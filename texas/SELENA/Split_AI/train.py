# -*- coding: utf-8 -*-
import argparse
import os
import random
import sys
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
from texas import TexasClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_train(train_data, train_labels, model, criterion, optimizer, batch_size):
    model.train() 

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t =  int(np.ceil(len(train_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min(len(train_data), (batch_ind+1)*batch_size)
        features = train_data[batch_ind*batch_size: end_idx]
        labels = train_labels[batch_ind*batch_size: end_idx]

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.long)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)

def split_test(test_data, test_labels, model, criterion, batch_size):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()    

    len_t =  int(np.ceil(len(test_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)

        features = test_data[batch_ind*batch_size: end_idx]
        labels = test_labels[batch_ind*batch_size: end_idx]

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.long)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)


def split_save_checkpoint(state, is_best, acc, split_name, checkpoint):
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
    parser = argparse.ArgumentParser(description = 'Setting for Texas datataset')
    parser.add_argument('--K', type = int, default = 25, help = 'total sub-models in split-ai')
    parser.add_argument('--L', type = int, default = 10, help = 'non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--split_epochs', type = int, default = 20, help = 'training epochs for each single model in split-ai')
    parser.add_argument('--print_epoch_splitai', type = int, default = 5, help = 'print splitai single model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    split_epochs = args.split_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    print_epoch_splitai = args.print_epoch_splitai
    load_name = str(split_model) + '_' + str(non_model)

    DATASET_PATH = os.path.join(root_dir, 'texas',  'data')
    checkpoint_path = os.path.join(root_dir, 'texas', 'checkpoints', 'K_L', load_name)
    checkpoint_path_splitai = os.path.join(checkpoint_path, 'split_ai')
    checkpoint_path_selena = os.path.join(checkpoint_path , 'selena')
    print(checkpoint_path, checkpoint_path_selena)
    
    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'defender', 'tr_data.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'defender', 'te_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))
    train_data = np.concatenate((train_data_tr_attack, train_data_te_attack), axis = 0)
    train_label = np.concatenate((train_label_tr_attack, train_label_te_attack), axis = 0)
    test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'))
    test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))
    ref_data = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'))
    ref_label = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'))
    all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))

    #print first 20 labels for each subset, for checking with other experiments
    print(train_label_tr_attack[:20])
    print(train_label_te_attack[:20])
    print(test_label[:20])
    print(ref_label[:20])

    best_acc = 0.0

    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    split_train_accs = []
    split_test_accs = []
    test_accs = []

    for i in range(split_model):
        split_best_acc = 0
        is_train_acc = 0
        is_test_acc = 0
        s_test_acc = 0
        best_epoch = 0

        model = TexasClassifier().to(device, torch.float)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        split_train_data_list = []
        split_train_label_list = []
        split_test_data_list = []
        split_test_label_list = []

        for ind in range(len(train_data)):
            tmp_ind = train_data[ind, -non_model:]
            if i not in tmp_ind:
                split_train_data_list.append(train_data[ind,:-non_model])
                split_train_label_list.append(train_label[ind])
            else:
                split_test_data_list.append(train_data[ind,:-non_model])
                split_test_label_list.append(train_label[ind])

        split_train_data = np.array(split_train_data_list)
        split_train_label = np.array(split_train_label_list)
        split_test_data = np.array(split_test_data_list)
        split_test_label = np.array(split_test_label_list)

        split_test_data_tensor = torch.from_numpy(split_test_data).type(torch.FloatTensor)
        split_test_label_tensor = torch.from_numpy(split_test_label).type(torch.LongTensor)

        #print first 50 labels for each subset i in splitai, for later checking
        print("split model: {:d},# of data: {:d}".format(i, len(split_train_data)))
        print(split_train_label[:50])
    
        for epoch in range(split_epochs):
            r= np.arange(len(split_train_data))
            np.random.shuffle(r)
            split_train_data = split_train_data[r]
            split_train_label = split_train_label[r]

            split_train_data_tensor = torch.from_numpy(split_train_data).type(torch.FloatTensor)
            split_train_label_tensor = torch.from_numpy(split_train_label).type(torch.LongTensor)
   
            _, split_train_acc = split_train(split_train_data_tensor, split_train_label_tensor, model, criterion, optimizer, batch_size)
            _, split_is_train_acc = split_test(split_train_data_tensor, split_train_label_tensor, model, criterion, batch_size)
            _, split_is_test_acc = split_test(split_test_data_tensor, split_test_label_tensor, model, criterion, batch_size)    
            _, split_test_acc = split_test(test_data_tensor,test_label_tensor, model, criterion, batch_size)


            split_is_best = split_test_acc >split_best_acc
    
            if split_is_best:
                is_train_acc = split_is_train_acc
                is_test_acc = split_is_test_acc
                s_test_acc = split_test_acc
                best_epoch = epoch + 1

            split_best_acc = max(split_test_acc, split_best_acc)
            
            if (epoch+1)%print_epoch_splitai== 0:
                print ('Epoch: [{:d} | {:d}]: train acc:{:.4f}, split train/test acc: {:.4f}/{:.4f}, test acc: {:.4f}. '.format(epoch+1, split_epochs, split_train_acc, split_is_train_acc, split_is_test_acc, split_test_acc))
                sys.stdout.flush()
            if split_is_best:
               split_save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': split_test_acc,
                   'best_acc': split_best_acc,
                   'optimizer' : optimizer.state_dict(),
                }, split_is_best, split_best_acc, split_name = str(i), checkpoint = checkpoint_path_splitai)
    
        print("model {:d} final saved epoch {:d}".format(i, best_epoch))
        split_train_accs.append(is_train_acc)
        split_test_accs.append(is_test_acc)
        test_accs.append(s_test_acc)
    print("For a single model, split train accuracy: {:.4f}, split test accuracy: {:.4f}, test accuracy: {:.4f}".format(np.mean(split_train_accs), np.mean(split_test_accs), np.mean(test_accs)))

if __name__ == '__main__':
    main()
