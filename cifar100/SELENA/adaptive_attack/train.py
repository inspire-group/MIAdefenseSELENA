import argparse
import os
import sys
import yaml
import random
import numpy as np

import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
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

def split_train(trainloader, model, criterion, optimizer, epoch, args, warmup_scheduler):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):
        if epoch <= args.warmup:
            warmup_scheduler.step()

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute gradient and do SGD step        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top5.update(prec5.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)

def split_test(testloader, model, criterion):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top5.update(prec5.item()/100.0, inputs.size()[0])
    return (losses.avg, top1.avg)


def split_save_checkpoint(state, is_best, acc, split_name, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(os.path.join(checkpoint, split_name)):
        mkdir_p(os.path.join(checkpoint, split_name))
    if is_best:
        filepath = os.path.join(checkpoint, split_name, 'model_best.pth.tar')
        if os.path.exists(filepath):
            tmp_ckpt = torch.load(filepath)
            best_acc = tmp_ckpt['best_acc']
            if best_acc > acc:
                return
        torch.save(state, filepath)

def main():
    parser = argparse.ArgumentParser(description='setting for cifar100')
    parser.add_argument('--K', type=int, default=25, help='total sub-models in split-ai')
    parser.add_argument('--L', type=int, default=10, help='non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')
    parser.add_argument('--split_epochs', type=int, default=1, help='training epochs for each single model in split-ai')
    parser.add_argument('--print_epoch_attack', type=int, default=50, help='print nn attack accuracy stats per print_epoch during nn training')
    parser.add_argument('--print_epoch_splitai', type=int, default=5, help='print splitai single model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--warmup', type=int, default=1, help='warm up epochs')
    parser.add_argument('--num_worker', type=int, default=1, help='number workers')
    parser.add_argument('--num_class', type=int, default=100, help='num class')
    parser.add_argument('--known_ratio', type=float, default=0.5, help='known ratio of member samples by attacker')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    split_epochs = args.split_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    print_epoch_attack = args.print_epoch_attack
    print_epoch_splitai = args.print_epoch_splitai
    load_name = str(split_model) + '_' + str(non_model)
    warmup = args.warmup
    num_worker = args.num_worker

    DATASET_PATH = os.path.join(root_dir, 'cifar100',  'data')
    checkpoint_path = os.path.join(root_dir, 'cifar100', 'checkpoints', 'K_L', load_name)
    checkpoint_path_shadow = os.path.join(checkpoint_path, 'shadow', str(args.known_ratio))
    checkpoint_path_selena = os.path.join(checkpoint_path, 'selena')
    print(checkpoint_path_shadow, checkpoint_path_selena)

    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'attacker', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'attacker', 'te_label.npy'))
    full_train_data = np.concatenate((train_data_tr_attack, train_data_te_attack), axis = 0)
    full_train_label = np.concatenate((train_label_tr_attack, train_label_te_attack), axis = 0)
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

    len_train_data = len(full_train_label)
    known_num = int(args.known_ratio*len_train_data)
    attack_know_data = full_train_data[:known_num, :]
    attack_know_label = full_train_label[:known_num]
    attack_unknow_data = full_train_data[known_num:, :]
    attack_unknow_label = full_train_label[known_num:]

    testset = Cifardata(test_data, test_label, transform_test)
    refset = Cifardata(ref_data, ref_label, transform_test)

    trset = Cifardata(train_data_tr_attack, train_label_tr_attack, transform_test)
    teset = Cifardata(train_data_te_attack, train_label_te_attack, transform_test)
    alltestset = Cifardata(all_test_data, all_test_label, transform_test)

    trloader = torch.utils.data.DataLoader(trset, batch_size = batch_size, shuffle = False, num_workers=num_worker)
    teloader = torch.utils.data.DataLoader(teset, batch_size = batch_size, shuffle = False, num_workers = num_worker)
    alltestloader = torch.utils.data.DataLoader(alltestset, batch_size = batch_size, shuffle = False, num_workers = num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    criterion = (nn.CrossEntropyLoss()).to(device, torch.float)

    split_test_accs = []
    for i in range(split_model):
        split_best_acc = 0
        saved_epoch = 0

        model_t = resnet18()
        model = ModelwNorm(model_t)
        model = model.to(device, torch.float)
        #model = nn.DataParallel(model).to(device, torch.float)

        split_train_data_list = []
        split_train_label_list = []
        for ind in range(len(attack_know_data)):
            tmp_ind = attack_know_label[ind, -non_model:]
            if i not in tmp_ind:
                split_train_data_list.append(attack_know_data[ind])
                split_train_label_list.append(attack_know_label[ind,0])
        split_train_data = np.array(split_train_data_list)
        split_train_label = np.array(split_train_label_list)
        print("split model: {:d},# of data: {:d}".format(i, len(split_train_data)))
        print(split_train_label[:20])

        split_trainset = Cifardata(split_train_data, split_train_label, transform_train)
        split_trainloader = torch.utils.data.DataLoader(split_trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

        iter_per_epoch = len(split_trainloader)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [60, 120, 160], gamma=0.2) #learning rate decay
        warmup_scheduler = WarmUpLR(optimizer,  iter_per_epoch * warmup)

        for epoch in range(1, split_epochs+1):
            if epoch > 1:
                train_scheduler.step(epoch)

            _, split_train_acc = split_train(split_trainloader, model, criterion, optimizer, epoch, args, warmup_scheduler)
            _, split_test_acc = split_test(testloader, model, criterion)

            split_is_best = split_test_acc >split_best_acc
            split_best_acc = max(split_test_acc, split_best_acc)
            if split_is_best:
                saved_epoch = epoch

            if (epoch)%print_epoch_splitai == 0:
                print ('Epoch: [{:d} | {:d}]: train acc:{:.4f}, test acc: {:.4f}. '.format(epoch, split_epochs, split_train_acc, split_test_acc))
            if split_is_best:
               split_save_checkpoint({
                        'epoch': epoch ,
                        'state_dict': model.state_dict(),
                        'acc': split_test_acc,
                        'best_acc': split_best_acc,
                        'optimizer' : optimizer.state_dict(),
                    }, split_is_best, split_best_acc, split_name = str(i), checkpoint = checkpoint_path_shadow, filename='Depoch%d.pth.tar'%(epoch))
            sys.stdout.flush()

        print("model {:d} final saved epoch {:d}: {:.4f}".format(i, saved_epoch, split_best_acc))
        split_test_accs.append(split_best_acc)

    print("For a single model, test accuracy: {:.4f}".format(np.mean(split_test_accs)))

if __name__ == '__main__':
    main()
