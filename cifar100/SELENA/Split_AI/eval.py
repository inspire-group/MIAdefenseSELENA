import argparse
import os
import sys
import yaml
import random
import numpy as np

import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
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

def splitai_test(testloader, model, criterion, len_data, ckpt_path, non_model_indices_all, mode, args):
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

            with torch.no_grad():
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
                rand_ind = np.random.randint(non_model_indices_all.shape[0])

                if non_model == 1:
                    outputs_np[ind,:] = tmp_outputs_np[non_model_indices_all[rand_ind, 1:].astype(np.int32), ind, :]
                else:                
                    outputs_np[ind,:] = np.mean(tmp_outputs_np[non_model_indices_all[rand_ind, 1:].astype(np.int32), ind, :], axis = 0)


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


def main():
    parser = argparse.ArgumentParser(description='setting for cifar100')
    parser.add_argument('--K', type=int, default=25, help='total sub-models in split-ai')
    parser.add_argument('--L', type=int, default=10, help='non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')
    parser.add_argument('--print_epoch_splitai', type=int, default=5, help='print splitai single model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
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

    trloader = torch.utils.data.DataLoader(trset, batch_size=batch_size, shuffle = False, num_workers=num_worker)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle = False, num_workers = num_worker)
    alltestloader = torch.utils.data.DataLoader(alltestset, batch_size=batch_size, shuffle = False, num_workers = num_worker)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    traintestloader = torch.utils.data.DataLoader(traintestset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)


    original_train_label = train_label.copy()

    criterion = (nn.CrossEntropyLoss()).to(device, torch.float)

    net_1 = resnet18()
    net = ModelwNorm(net_1)

    print("Attack Training: # of train data: {:d}, # of ref data: {:d}".format(int(len(train_data_tr_attack)), len(ref_data)))
    print("Attack Testing: # of train data: {:d}, # of test data: {:d}".format(int(len(train_data_te_attack)), len(test_data)))


    print("training set")
    train_loss, infer_train_conf = splitai_test(traintestloader, net, criterion, len(traintestset), checkpoint_path_splitai, original_train_label, train_mode, args)
    train_acc, train_conf = print_acc_conf(infer_train_conf, train_label[:,0])
    print("all test set")
    all_test_loss, infer_all_test_conf = splitai_test(alltestloader, net, criterion, len(alltestset), checkpoint_path_splitai, original_train_label, test_mode, args)
    all_test_acc, all_test_conf = print_acc_conf(infer_all_test_conf, all_test_label)
    print("tr set")
    tr_loss, infer_train_conf_tr = splitai_test(trloader, net, criterion, len(trset), checkpoint_path_splitai, original_train_label, train_mode, args)
    tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack[:,0])
    print("te set")
    te_loss, infer_train_conf_te = splitai_test(teloader, net, criterion, len(teset), checkpoint_path_splitai, original_train_label, train_mode, args)
    te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack[:,0])
    print("test set")
    test_loss, infer_test_conf = splitai_test(testloader, net, criterion, len(testset), checkpoint_path_splitai, original_train_label, test_mode, args)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference set")
    ref_loss, infer_ref_conf = splitai_test(refloader, net, criterion, len(refset), checkpoint_path_splitai, original_train_label, test_mode, args)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    print("For comparison on splitai output")
    print("avg acc  on train/all test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_acc, all_test_acc ,tr_acc, te_acc, test_acc, ref_acc))
    print("avg conf on train/all_test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_conf, all_test_conf, tr_conf, te_conf, test_conf, ref_conf))

    system_attack(infer_train_conf_tr, train_label_tr_attack[:,0], infer_train_conf_te, train_label_te_attack[:,0], infer_ref_conf, ref_label, infer_test_conf, test_label, num_class,attack_epochs=attack_epochs,batch_size=256)

if __name__ == '__main__':
    main()
