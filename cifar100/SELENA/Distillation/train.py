import argparse
import os
import sys
import shutil
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

config_file = './../../../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'attack'))
sys.path.append(os.path.join(src_dir, 'models'))
from dsq_attack import system_attack
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from resnet import resnet18
from cifar_utils import transform_train, transform_test, Cifardata, DistillCifardata, WarmUpLR, ModelwNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splitai_test(testloader, model, criterion, len_data, ckpt_path, non_model_all_indices, mode, args):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()

    num_class = args.num_class
    batch_size = args.batch_size
    split_model = args.K
    non_model = args.L

    infer_np = np.zeros((len_data, num_class))
    cnt = np.zeros(split_model)
    corr = np.zeros(split_model)
    conf = np.zeros(split_model)

    for batch_ind, (inputs, targets) in enumerate(testloader):
        # compute output
        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs_np = np.zeros((inputs.shape[0], num_class))
        outputs_conf = np.zeros((split_model, inputs.shape[0]))
        tmp_outputs_np = np.zeros((split_model, inputs.shape[0], num_class))

        for model_ind in range(split_model):
            ckpt = torch.load(os.path.join(ckpt_path, str(model_ind), 'model_best.pth.tar'))
            model.load_state_dict(ckpt['state_dict'])
            model = model.to(device,torch.float)
            model.eval()

            tmp_outputs = (F.softmax(model(inputs),dim=1)).detach().cpu().numpy()
            tmp_outputs_np[model_ind,:,:] = tmp_outputs

            outputs_conf[model_ind,:] = np.max(tmp_outputs, axis = 1)
        
        temp = np.zeros(split_model)
        for ind in range(inputs.shape[0]):

            if mode == 1:
                if non_model == 1:
                    outputs_np[ind,:] = tmp_outputs_np[(targets[ind, 1].detach().cpu().numpy().astype(np.int32)), ind, :]
                else:                
                    outputs_np[ind,:] = np.mean(tmp_outputs_np[(targets[ind, 1:].detach().cpu().numpy().astype(np.int32)), ind, :], axis = 0)
            
            elif mode ==2:
                rand_ind = np.random.randint(non_model_all_indices.shape[0])

                if non_model == 1:
                    outputs_np[ind,:] = tmp_outputs_np[non_model_all_indices[rand_ind, 1:].astype(np.int32), ind, :]
                else:                
                    outputs_np[ind,:] = np.mean(tmp_outputs_np[non_model_all_indices[rand_ind, 1:].astype(np.int32), ind, :], axis = 0)


            tmp_rank = np.argsort(outputs_conf[:,ind])
            if mode == 1:
                for kidx in range(split_model):
                    if tmp_rank[kidx] in targets[ind, -non_model:]:
                        cnt[kidx] = cnt[kidx] + 1

                for model_ind in range(split_model):
                    if np.argmax(tmp_outputs_np[tmp_rank[model_ind],ind,:]) == int(targets[ind, 0]):
                        corr[model_ind] = corr[model_ind] + 1
                    temp[model_ind] = temp[model_ind] + outputs_conf[tmp_rank[model_ind], ind]

            if mode == 2:
                for model_ind in range(split_model):
                    if np.argmax(tmp_outputs_np[tmp_rank[model_ind],ind,:]) == int(targets[ind]):
                        corr[model_ind] = corr[model_ind] + 1
                    temp[model_ind] = temp[model_ind] + outputs_conf[tmp_rank[model_ind], ind]


        for model_ind in range(split_model):
            conf[model_ind] = conf[model_ind] + temp[model_ind]
        
        infer_np[batch_ind*batch_size: batch_ind*batch_size+inputs.shape[0]] = outputs_np
        outputs = torch.from_numpy(outputs_np).to(device, torch.float)        
        if mode == 1:
            loss = criterion(outputs, targets[:,0])
        if mode == 2:
            loss = criterion(outputs, targets)

    print(len_data)
    if mode == 1:
        for ind in range(split_model):
            print("{:d} least confidence : matches as non-training data {:d}/{:.4f}. corr: {:d}/{:.4f}. conf_avg: {:.4f}".format(ind+1, int(cnt[ind]), cnt[ind]*1.0/len_data, int(corr[ind]), corr[ind]*1.0/len_data, conf[ind]/len_data))
    elif mode == 2:
        for ind in range(split_model):
            print("{:d} least confidence: corr: {:d}/{:.4f}. conf_avg: {:.4f}".format(ind+1, int(corr[ind]), corr[ind]*1.0/len_data, conf[ind]/len_data))
    return (losses.avg, infer_np)



def distill_train(distilltrainloader, model, criterion, optimizer, epoch, warmup_scheduler1, args):
    model.train() 

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_ind, (features, confs, conf_labels, labels) in enumerate(distilltrainloader):
        if epoch <= args.warmup:
            warmup_scheduler1.step()

        inputs = features.to(device, torch.float)
        targets = confs.to(device, torch.float)
        target_labels = conf_labels.to(device, torch.long)

        # compute output
        outputs = model(inputs)

        one_hot_tr = torch.zeros(inputs.size()[0], outputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, target_labels.view([-1,1]), 1)

        #loss = criterion(outputs, target_labels)
        loss = (-torch.sum(targets*torch.log(F.softmax(outputs,dim=1))))/inputs.shape[0]

        # measure accuracy and record loss
        prec1, _ = accuracy(outputs.data, target_labels.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)

def distill_test(distilltestloader, model, criterion, len_data):
    model.eval()

    losses = AverageMeter()

    cnt1 = 0
    cnt1_1 = 0
    cnt = 0

    for batch_ind, (features, confs, conf_labels, labels) in enumerate(distilltestloader):


        labels1 = labels.to(device, torch.long)
        inputs = features.to(device, torch.float)
        targets = confs.to(device, torch.float)
        target_labels = conf_labels.to(device, torch.long)

        outputs = model(inputs)
        outputs_np = outputs.detach().cpu().numpy()
        outputs_np_ind = np.argmax(outputs_np, axis = 1)

        loss = criterion(outputs, target_labels)

        for ind in range(inputs.shape[0]):
            if target_labels[ind]== labels1[ind] and outputs_np_ind[ind] ==labels1[ind]:
                cnt = cnt + 1
            if outputs_np_ind[ind] == target_labels[ind]:
                cnt1 = cnt1 +1
            if outputs_np_ind[ind] == labels1[ind]:
                cnt1_1 = cnt1_1 + 1

        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, cnt1/len_data, cnt1_1/len_data, cnt/len_data)


def selena_test(testloader, model, criterion, batch_size, len_data):
    model.eval()

    losses = AverageMeter()
    infer_np = np.zeros((len_data, num_class))

    for batch_ind, (features, labels) in enumerate(testloader):

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.long)
        outputs = model(inputs)

        infer_np[batch_ind*batch_size:batch_ind*batch_size + inputs.shape[0]] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()

        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, infer_np)


def distill_save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
        
    if is_best:
        torch.save(state, os.path.join(checkpoint, 'model_best.pth.tar'))

def main():
    parser = argparse.ArgumentParser(description='setting for cifar100')
    parser.add_argument('--K', type=int, default=25, help='total sub-models in split-ai')
    parser.add_argument('--L', type=int, default=10, help='non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')
    parser.add_argument('--classifier_epochs', type=int, default=200, help='classifier epochs in distillation')
    parser.add_argument('--print_epoch_splitai', type=int, default=5, help='print splitai single model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--warmup', type=int, default=1, help='warm up epochs')
    parser.add_argument('--num_worker', type=int, default=1, help='number workers')
    parser.add_argument('--num_class', type=int, default=100, help='num class')


    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    classifier_epochs = args.classifier_epochs
    print_epoch_splitai = args.print_epoch_splitai
    load_name = str(split_model) + '_' + str(non_model)
    warmup = args.warmup
    num_worker = args.num_worker

    train_mode = 1
    test_mode = 2

    DATASET_PATH = os.path.join(root_dir, 'cifar100',  'data')
    checkpoint_path = os.path.join(root_dir, 'cifar100', 'checkpoints', 'K_L', load_name)
    checkpoint_path_splitai = os.path.join(checkpoint_path, 'split_ai')
    checkpoint_path_selena = os.path.join(checkpoint_path, 'selena')
    print(checkpoint_path, checkpoint_path_selena)

    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'defender', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'defender', 'te_label.npy'))
    train_data = np.concatenate((train_data_tr_attack, train_data_te_attack), axis = 0)
    train_label = np.concatenate((train_label_tr_attack, train_label_te_attack), axis = 0)
    test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'))
    test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))
    ref_data = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'))
    ref_label = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'))
    all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))

    #print first 20 labels for each subset, for checking with other experiments
    print(train_label_tr_attack[:20, 0])
    print(train_label_te_attack[:20, 0])
    print(test_label[:20])
    print(ref_label[:20])

    trainset = Cifardata(train_data, train_label, transform_train)
    traintestset = Cifardata(train_data, train_label, transform_test)
    testset = Cifardata(test_data, test_label, transform_test)
    refset = Cifardata(ref_data, ref_label, transform_test)

    trset = Cifardata(train_data_tr_attack, train_label_tr_attack, transform_test)
    teset = Cifardata(train_data_te_attack, train_label_te_attack, transform_test)
    alltestset = Cifardata(all_test_data, all_test_label, transform_test)

    args.batch_size=2048
    traintestloader = torch.utils.data.DataLoader(traintestset, batch_size=args.batch_size,shuffle=False,num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=False,num_workers=num_worker)


    original_train_label = train_label.copy()

    criterion = (nn.CrossEntropyLoss()).to(device, torch.float)

    net_1 = resnet18()
    net = ModelwNorm(net_1)

    print("Attack Training: # of train data: {:d}, # of ref data: {:d}".format(int(len(train_data_tr_attack)), len(ref_data)))
    print("Attack Testing: # of train data: {:d}, # of test data: {:d}".format(int(len(train_data_te_attack)), len(test_data)))


    print("training sets")
    train_loss, infer_train_conf = splitai_test(traintestloader, net, criterion, len(traintestset), checkpoint_path_splitai, original_train_label, train_mode, args)
    train_acc, train_conf = print_acc_conf(infer_train_conf, train_label[:, 0])
    test_loss, infer_test_conf = splitai_test(testloader, net, criterion, len(testset), checkpoint_path_splitai, original_train_label, test_mode, args)
    test_acc, test_conf= print_acc_conf(infer_test_conf, test_label)
    args.batch_size = batch_size


    infer_train_label = np.argmax(infer_train_conf, axis = 1)
    infer_test_label = np.argmax(infer_test_conf, axis = 1)

    distilltrainset = DistillCifardata(train_data, infer_train_conf, infer_train_label, train_label[:,0], transform_train)
    distilltraintestset = DistillCifardata(train_data, infer_train_conf, infer_train_label, train_label[:,0], transform_test)
    distilltestset = DistillCifardata(test_data, infer_test_conf, infer_test_label, test_label, transform_test)

    distilltrainloader = torch.utils.data.DataLoader(distilltrainset, batch_size=batch_size, shuffle=True, num_workers= num_worker)
    distilltraintestloader = torch.utils.data.DataLoader(distilltraintestset, batch_size=batch_size, shuffle=False, num_workers= num_worker)
    distilltestloader = torch.utils.data.DataLoader(distilltestset, batch_size=batch_size, shuffle=False, num_workers= num_worker)

    print("total data for distillation: {:d}".format(len(distilltraintestset)))


    net1_t = resnet18()
    net1 = ModelwNorm(net1_t)
    net1 = net1.to(device, torch.float)

    criterion1 = nn.CrossEntropyLoss().to(device, torch.float)

    iter_per_epoch = len(distilltrainloader)
    optimizer1 = optim.SGD(net1.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones= [60, 120, 160], gamma=0.2) #learning rate decay
    warmup_scheduler1 = WarmUpLR(optimizer1,  iter_per_epoch * warmup)

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, classifier_epochs+1):
        if epoch > 1:
            train_scheduler1.step(epoch)

        train_loss, train_acc = distill_train(distilltrainloader, net1, criterion1, optimizer1, epoch, warmup_scheduler1, args)

        train_test_loss, train_test_acc, train_test_acc1, train_test_acc2 = distill_test(distilltraintestloader, net1, criterion1, len(distilltrainset))

        test_loss, test_acc, test_acc1, test_acc2 = distill_test(distilltestloader, net1, criterion1, len(distilltestset))

        # save model
        is_best = test_acc1>best_acc
        if is_best:
            best_epoch = epoch
        best_acc = max(test_acc1, best_acc)

        distill_save_checkpoint({
                    'epoch': epoch ,
                    'state_dict': net1.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer1.state_dict(),
                }, is_best, checkpoint=checkpoint_path_selena, filename='Depoch%d.pth.tar'%(epoch))

        print('Epoch: [{:d} | {:d}]: loss: training/train/test: {:.4f}/{:.4f}/{:.4f}. distll label training acc: {:.4f}. acc: train/test: {:.4f}/{:.4f}/{:.4f}|{:.4f}/{:.4f}/{:.4f}.[soft_label|true label|intersect]'.format(epoch, classifier_epochs, train_loss, train_test_loss, test_loss, train_acc, train_test_acc, train_test_acc1, train_test_acc2, test_acc, test_acc1, test_acc2))
        sys.stdout.flush()
    print("Final saved epoch {:d} acc: {:.4f}".format(best_epoch, best_acc))

if __name__ == '__main__':
    main()
