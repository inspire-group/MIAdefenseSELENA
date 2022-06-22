# -*- coding: utf-8 -*-
import argparse
import os
import sys
import yaml
import random
import numpy as np

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
from purchase import PurchaseClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splitai_test(test_data, test_labels, model, criterion, feature_len, original_train_data, ckpt_path, mode, args, is_print = False):
    batch_size = args.batch_size
    num_class = args.num_class
    split_model = args.K
    non_model = args.L

    len_t =  int(np.ceil(len(test_data)/batch_size))

    infer_np = np.zeros((len(test_data), num_class))

    cnt = np.zeros(split_model)
    corr = np.zeros(split_model)
    conf = np.zeros(split_model)

    for batch_ind in range(len_t):

        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        features = test_data[batch_ind*batch_size: end_idx, :feature_len]
        labels = test_labels[batch_ind*batch_size: end_idx]

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.long)

        # compute output
        outputs_np = np.zeros((inputs.shape[0], num_class))
        outputs_conf = np.zeros((split_model, inputs.shape[0]))
        tmp_outputs_np = np.zeros((split_model, inputs.shape[0], num_class))

        for model_ind in range(split_model):
            ckpt = torch.load(os.path.join(ckpt_path, str(model_ind), 'model_best.pth.tar'))
            model.load_state_dict(ckpt['state_dict'])
            model = model.to(device,torch.float)
            model.eval()

            tmp_outputs = (F.softmax(model(inputs), dim=1)).detach().cpu().numpy()
            tmp_outputs_np[model_ind,:,:] = tmp_outputs
            outputs_conf[model_ind,:] = np.max(tmp_outputs, axis = 1)

        temp = np.zeros(split_model)
        for ind in range(inputs.shape[0]):
            if mode == 1:
                if non_model == 1:
                    outputs_np[ind,:] = tmp_outputs_np[(test_data[batch_ind*batch_size+ind, -non_model:].detach().numpy().astype(np.int32)), ind, :]
                else:                
                    outputs_np[ind,:] = np.mean(tmp_outputs_np[(test_data[batch_ind*batch_size+ind, -non_model:].detach().numpy().astype(np.int32)), ind, :], axis = 0)
            
            elif mode ==2:
                rand_ind = np.random.randint(original_train_data.shape[0])

                if non_model == 1:
                    outputs_np[ind,:] = tmp_outputs_np[original_train_data[rand_ind, -non_model:].astype(np.int32), ind, :]
                else:                
                    outputs_np[ind,:] = np.mean(tmp_outputs_np[original_train_data[rand_ind, -non_model:].astype(np.int32), ind, :], axis = 0)

            tmp_rank = np.argsort(outputs_conf[:,ind])
            if mode == 1:

                for kidx in range(split_model):
                    if tmp_rank[kidx] in test_data[batch_ind*batch_size + ind, -non_model:]:
                        cnt[kidx] = cnt[kidx] + 1

            for model_ind in range(split_model):
                if np.argmax(tmp_outputs_np[tmp_rank[model_ind],ind,:]) == int(test_labels[batch_ind*batch_size+ind]):
                    corr[model_ind] = corr[model_ind] + 1
                temp[model_ind] = temp[model_ind] + outputs_conf[tmp_rank[model_ind], ind]

        for model_ind in range(split_model):
            conf[model_ind] = conf[model_ind] + temp[model_ind]/inputs.shape[0]
        
        infer_np[batch_ind*batch_size: end_idx] = outputs_np

    print(len(test_data))
    if mode == 1:
        for ind in range(split_model):
            print("{:d} least confidence : matches as non-training data {:d}/{:.4f}. corr: {:d}/{:.4f}. conf_avg: {:.4f}".format(ind+1, int(cnt[ind]), cnt[ind]*1.0/len(test_data), int(corr[ind]), corr[ind]*1.0/len(test_data), conf[ind]/len_t))
    elif mode == 2:
        for ind in range(split_model):
            print("{:d} least confidence: corr: {:d}/{:.4f}. conf_avg: {:.4f}".format(ind+1, int(corr[ind]), corr[ind]*1.0/len(test_data), conf[ind]/len_t))

    return infer_np


def distill_train(train_data, train_labels, model, criterion, optimizer, batch_size):
    model.train() 

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t =  int(np.ceil(len(train_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min(len(train_data), (batch_ind+1)*batch_size)
        features = train_data[batch_ind*batch_size: end_idx]
        labels = train_labels[batch_ind*batch_size: end_idx]

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.float)
        target_labels = torch.argmax(targets, dim =1)

        # compute output
        outputs = model(inputs)

        one_hot_tr = torch.zeros(inputs.size()[0], outputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, target_labels.view([-1,1]), 1)


        loss = (-torch.sum(targets*torch.log(F.softmax(outputs, dim=1))))/(end_idx-batch_ind*batch_size)
        #criterion(outputs, target_labels)

        # measure accuracy and record loss
        prec1, _ = accuracy(outputs.data, target_labels.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)


def distill_test(test_data, test_confs, infer_test_labels, test_labels, model, criterion, batch_size):
    model.eval()

    losses = AverageMeter()
    losses1 = AverageMeter()
    top1 = AverageMeter()
    top1_1 = AverageMeter()

    cnt = 0
    len_t =  int(np.ceil(len(test_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)

        features = test_data[batch_ind*batch_size: end_idx]
        labels = test_confs[batch_ind*batch_size: end_idx]

        labels1 = test_labels[batch_ind*batch_size:end_idx].to(device, torch.long)

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.float)
        target_labels = infer_test_labels[batch_ind*batch_size:end_idx].to(device, torch.long)

        outputs = model(inputs)
        outputs_np = outputs.detach().cpu().numpy()
        outputs_np_ind = np.argmax(outputs_np, axis = 1)
        
        loss = (-torch.sum((labels.to(device, torch.float))*torch.log(F.softmax(outputs,dim=1))))/(end_idx-batch_ind*batch_size)
#        loss = criterion(outputs, target_labels)

        for ind in range(inputs.shape[0]):
            if target_labels[ind]== labels1[ind] and outputs_np_ind[ind] ==labels1[ind]:
                cnt = cnt + 1

        prec1, _ = accuracy(outputs.data, target_labels.data, topk=(1, 5))
        prec1_1, _ = accuracy(outputs.data, labels1.data, topk=(1, 5))        
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top1_1.update(prec1_1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg, top1_1.avg, cnt/len(test_data))

def distill_save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    if is_best:
        torch.save(state, os.path.join(checkpoint, 'model_best.pth.tar'))

def main():
    parser = argparse.ArgumentParser(description='Setting for Purchase datataset')
    parser.add_argument('--K', type=int, default=25, help='total sub-models in split-ai')
    parser.add_argument('--L', type=int, default=10, help='non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')
    parser.add_argument('--classifier_epochs', type=int, default=60, help='attack epochs in NN attack')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_class', type=int, default=100, help='num class')
    parser.add_argument('--config_file', type=str, default='./../../../env.yml', help='configuration for src file directory and data/ckpts directory')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    classifier_epochs = args.classifier_epochs
    config_file = args.config_file
    load_name = str(split_model) + '_' + str(non_model)

    DATASET_PATH = os.path.join(root_dir, 'purchase',  'data')
    checkpoint_path = os.path.join(root_dir, 'purchase', 'checkpoints', 'K_L', load_name)
    checkpoint_path_splitai = os.path.join(checkpoint_path, 'split_ai')
    checkpoint_path_selena = os.path.join(checkpoint_path, 'selena')
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
    feature_len = ref_data.shape[1]

    train_mode = 1
    test_mode = 2
    
    raw_train_data = train_data.copy()

    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)
    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    net = PurchaseClassifier()
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    print("training set")
    infer_train_conf = splitai_test(train_data_tensor, train_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, train_mode, args,True)
    train_acc, train_conf = print_acc_conf(infer_train_conf, train_label)
    print("test set")
    infer_test_conf = splitai_test(test_data_tensor, test_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, test_mode, args, True)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)

    infer_test_conf_tensor = torch.from_numpy(infer_test_conf).type(torch.FloatTensor)

    infer_train_conf_ind = np.argmax(infer_train_conf, axis = 1)

    soft_train_conf = infer_train_conf
    soft_train_data = train_data
    soft_train_data = soft_train_data[:, :-non_model]
    soft_train_label = train_label

    net1 = PurchaseClassifier().to(device, torch.float)
    criterion1 = nn.CrossEntropyLoss().to(device, torch.float)
    optimizer1 = optim.Adam(net1.parameters(), lr=0.001)#, weight_decay = 0.0001)

    r= np.arange(len(soft_train_data))
    np.random.shuffle(r)
    soft_train_data = soft_train_data[r]
    soft_train_label = soft_train_label[r]
    soft_train_conf = soft_train_conf[r]
    soft_train_data_tensor = torch.from_numpy(soft_train_data).type(torch.FloatTensor)
    soft_train_label_tensor = torch.from_numpy(soft_train_label).type(torch.LongTensor)
    soft_train_conf_tensor = torch.from_numpy(soft_train_conf).type(torch.FloatTensor)    


    soft_infer_train_label = np.argmax(soft_train_conf, axis = 1)
    infer_test_label = np.argmax(infer_test_conf, axis = 1)

    soft_infer_train_label_tensor = torch.from_numpy(soft_infer_train_label).type(torch.LongTensor)
    infer_test_label_tensor = torch.from_numpy(infer_test_label).type(torch.LongTensor)

    soft_train_accs = []
    soft_train_test_accs = []
    soft_test_accs = []
    
    print("distill training: total data: {:d}".format(len(soft_train_data)))

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(classifier_epochs):
    
        r= np.arange(len(soft_train_data))
        np.random.shuffle(r)
        soft_train_data = soft_train_data[r]
        soft_train_label = soft_train_label[r]
        soft_train_conf = soft_train_conf[r]
        soft_infer_train_label = soft_infer_train_label[r]

        soft_train_data_tensor = torch.from_numpy(soft_train_data).type(torch.FloatTensor)
        soft_train_label_tensor = torch.from_numpy(soft_train_label).type(torch.LongTensor)
        soft_train_conf_tensor = torch.from_numpy(soft_train_conf).type(torch.FloatTensor)
        soft_infer_train_label_tensor = torch.from_numpy(soft_infer_train_label).type(torch.LongTensor)

     
        train_loss, train_acc = distill_train(soft_train_data_tensor, soft_train_conf_tensor, net1, criterion1, optimizer1, batch_size)
        soft_train_accs.append(train_acc)

        train_test_loss, train_test_acc, train_test_acc1, train_test_acc2 = distill_test(soft_train_data_tensor, soft_train_conf_tensor, soft_infer_train_label_tensor, soft_train_label_tensor, net1, criterion1, batch_size)
        soft_train_test_accs.append(train_test_acc)

        test_loss, test_acc, test_acc1, test_acc2 = distill_test(test_data_tensor, infer_test_conf_tensor, infer_test_label_tensor, test_label_tensor, net1, criterion1, batch_size)
        soft_test_accs.append(test_acc)

        # save model
        is_best = test_acc1>best_acc
        best_acc = max(test_acc1, best_acc)

        if is_best:
            best_epoch = epoch + 1

        distill_save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net1.state_dict(),
                    'acc': test_acc1,
                    'best_acc': best_acc,
                    'optimizer' : optimizer1.state_dict(),
                }, is_best, checkpoint=checkpoint_path_selena, filename='Depoch%d.pth.tar'%(epoch+1))

        print('Epoch: [{:d} | {:d}]: loss: training/train/test: {:.4f}/{:.4f}/{:.4f}. soft label training acc: {:.4f}. acc: train/test: {:.4f}/{:.4f}/{:.4f}|{:.4f}/{:.4f}/{:.4f}.[soft_label|true label|intersect]'.format(epoch+1, classifier_epochs, train_loss, train_test_loss, test_loss, train_acc, train_test_acc, train_test_acc1, train_test_acc2, test_acc, test_acc1, test_acc2))
        sys.stdout.flush()
    print("final saved epoch: {:d}".format(best_epoch))

if __name__ == '__main__':
    main()