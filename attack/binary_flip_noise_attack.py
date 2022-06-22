# -*- coding: utf-8 -*-
import argparse
import os
import random
import numpy as np
import sys

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

from dsq_attack import system_attack
from utils import AverageMeter, print_acc_conf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
small_delta = 1e-30

def test(test_data, labels, model, criterion, batch_size):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    len_t =  int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
        
        loss = criterion(outputs, targets)
        losses.update(loss.data, inputs.size()[0])

    return (losses.avg, infer_np) #, logits_np)

def threshold_based_inference_attack(train_member_stat,train_member_label,train_nonmember_stat,train_nonmember_label,test_member_stat,test_member_label,test_nonmember_stat,test_nonmember_label, num_class=100,per_class=True):
    """
    train_member_stat: member samples for finding threshold
    train_nonmember_stat: nonmember samples for finding threshold
    test_member_stat: member samples for MIA
    test_nonmember_stat: nonmember samples for evaluation MIA
    Note: Both stats are assumed to behave like confidence values, i.e., higher is better. Negate the values if it behaves in the opposite way, e.g., for xe-loss, lower is better
    """
    #global threshold 
    list_all = np.concatenate((train_member_stat, train_nonmember_stat))
    max_gap = 0
    thre_chosen_g = 0
    list_all.sort()
    for thre in list_all:
        ratio1 = np.sum(train_member_stat>=thre)
        ratio2 = len(train_nonmember_stat)-np.sum(train_nonmember_stat>=thre)
        if ratio1+ratio2 > max_gap:
            max_gap = ratio1+ratio2
            thre_chosen_g = thre
    #evaluate global threshold
    ratio1 = np.sum(test_member_stat>=thre_chosen_g)
    ratio2 = len(test_nonmember_stat)-np.sum(test_nonmember_stat>=thre_chosen_g)
    global_MIA_acc = (ratio1+ratio2)/(len(test_member_stat)+len(test_nonmember_stat))

    if per_class == True:
        #per-class threshold
        thre_chosen_class = np.zeros(num_class)
        for i in range(num_class):
            train_member_stat_class = train_member_stat[train_member_label==i]
            train_nonmember_stat_class = train_nonmember_stat[train_nonmember_label==i]
            list_all_class = np.concatenate((train_member_stat_class, train_nonmember_stat_class))
            max_gap = 0
            thre_chosen = 0
            list_all_class.sort()
            for thre in list_all_class:
                ratio1 = np.sum(train_member_stat_class>=thre)
                ratio2 = len(train_nonmember_stat_class)-np.sum(train_nonmember_stat_class>=thre)
                if ratio1+ratio2 > max_gap:
                    max_gap = ratio1+ratio2
                    thre_chosen = thre
            thre_chosen_class[i] = thre_chosen
        #evaluate per class threshold
        ratio1 = np.sum(test_member_stat>=thre_chosen_class[test_member_label])
        ratio2 = len(test_nonmember_stat) - np.sum(test_nonmember_stat>=thre_chosen_class[test_nonmember_label])
        class_MIA_acc = (ratio1+ratio2)/(len(test_member_stat)+len(test_nonmember_stat))
        return max(global_MIA_acc, class_MIA_acc), global_MIA_acc, thre_chosen_g, class_MIA_acc, thre_chosen_class
    else:
        return global_MIA_acc, thre_chosen_g

def flip_noise_attack(net2, know_train_data, know_train_label, unknow_train_data, unknow_train_label, ref_data, ref_label, test_data, test_label, num_class, batch_size, flip_range, repeated=100, attack_epochs = 150, seed=123456789):
    net2 = net2.to(device,torch.float)
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    len1, len2 = min(len(know_train_label), len(ref_label)),min(len(unknow_train_label), len(test_label))

    know_train_data, know_train_label = know_train_data[:len1], know_train_label[:len1]
    ref_data, ref_label = ref_data[:len1], ref_label[:len1]
    unknow_train_data, unknow_train_label = unknow_train_data[:len2], unknow_train_label[:len2]
    test_data, test_label = test_data[:len2], test_label[:len2]

    print("\n\nEvaluating label-only attacks:", len(know_train_label), len(unknow_train_label), len(ref_label), len(test_label))
    print("batch_size", batch_size)
    print(know_train_label[:20])
    print(unknow_train_label[:20])
    print(test_label[:20])
    print(ref_label[:20])

    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)
    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    know_train_data_tensor = torch.from_numpy(know_train_data).type(torch.FloatTensor)
    know_train_label_tensor = torch.from_numpy(know_train_label).type(torch.LongTensor)
    unknow_train_data_tensor = torch.from_numpy(unknow_train_data).type(torch.FloatTensor)
    unknow_train_label_tensor = torch.from_numpy(unknow_train_label).type(torch.LongTensor)

    print("known train set (train_members)")
    _, know_infer_train_conf = test(know_train_data_tensor, know_train_label_tensor, net2, criterion, batch_size)
    know_acc, know_conf = print_acc_conf(know_infer_train_conf, know_train_label)
    print("unknown train set (test_members)")
    _loss, unknow_infer_train_conf= test(unknow_train_data_tensor, unknow_train_label_tensor, net2, criterion, batch_size)
    unknow_acc, unknow_conf = print_acc_conf(unknow_infer_train_conf, unknow_train_label)
    print("test set (test_nonmembers)")
    _, infer_test_conf = test(test_data_tensor, test_label_tensor, net2, criterion, batch_size)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference set (train_nonmembers)")
    _, infer_ref_conf = test(ref_data_tensor, ref_label_tensor, net2, criterion, batch_size)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    print("For comparison")
    print("avg acc  on know/unknow/ref/test set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_acc, unknow_acc, ref_acc, test_acc))
    print("avg conf on know/unknow/ref/test set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_conf, unknow_conf, ref_conf, test_conf))

    print("Double check for best direct single query attack")
    system_attack(know_infer_train_conf, know_train_label, unknow_infer_train_conf, unknow_train_label, infer_ref_conf, ref_label, infer_test_conf, test_label, num_class=num_class, attack_epochs=attack_epochs, batch_size=batch_size)

    best_attack_acc = 0
    np.random.seed(seed)
    for i in range(1, flip_range+1):
        print('\n', i)

        test_score = np.zeros((test_data.shape[0]))
        ref_score = np.zeros((ref_data.shape[0]))
        know_score = np.zeros((know_train_data.shape[0]))
        unknow_score = np.zeros((unknow_train_data.shape[0]))

        for k in range(repeated):
            noisy_test_data = test_data.copy()
            noisy_test_label = test_label.copy()

            for j in range(len(noisy_test_data)):
                r = np.arange(noisy_test_data.shape[1])
                np.random.shuffle(r)
                noisy_test_data[j, r[:i]] = 1 - noisy_test_data[j, r[:i]]

            noisy_test_data_tensor = torch.from_numpy(noisy_test_data).type(torch.FloatTensor)
            noisy_test_label_tensor = torch.from_numpy(noisy_test_label).type(torch.LongTensor)

            _, noisy_test_conf = test(noisy_test_data_tensor, noisy_test_label_tensor, net2, criterion, batch_size)
            test_score = test_score + (np.argmax(noisy_test_conf, axis = 1) == noisy_test_label)

            noisy_ref_data = ref_data.copy()
            noisy_ref_label = ref_label.copy()
            for j in range(len(noisy_ref_data)):
                r = np.arange(noisy_ref_data.shape[1])
                np.random.shuffle(r)
                noisy_ref_data[j, r[:i]] = 1 - noisy_ref_data[j, r[:i]]
            noisy_ref_data_tensor = torch.from_numpy(noisy_ref_data).type(torch.FloatTensor)
            noisy_ref_label_tensor = torch.from_numpy(noisy_ref_label).type(torch.LongTensor)
            _, noisy_ref_conf = test(noisy_ref_data_tensor, noisy_ref_label_tensor, net2, criterion, batch_size)
            ref_score = ref_score + (np.argmax(noisy_ref_conf, axis = 1) == noisy_ref_label)

            noisy_know_train_data = know_train_data.copy()
            noisy_know_train_label = know_train_label.copy()
            for j in range(len(noisy_know_train_data)):
                r = np.arange(noisy_know_train_data.shape[1])
                np.random.shuffle(r)
                noisy_know_train_data[j, r[:i]] = 1 - noisy_know_train_data[j, r[:i]]
            noisy_know_train_data_tensor = torch.from_numpy(noisy_know_train_data).type(torch.FloatTensor)
            noisy_know_train_label_tensor = torch.from_numpy(noisy_know_train_label).type(torch.LongTensor)

            _, noisy_know_train_conf = test(noisy_know_train_data_tensor, noisy_know_train_label_tensor, net2, criterion, batch_size)
            know_score = know_score + (np.argmax(noisy_know_train_conf, axis = 1)  == noisy_know_train_label)

            noisy_unknow_train_data = unknow_train_data.copy()
            noisy_unknow_train_label = unknow_train_label.copy()
            for j in range(len(noisy_unknow_train_data)):
                r = np.arange(noisy_unknow_train_data.shape[1])
                np.random.shuffle(r)
                noisy_unknow_train_data[j, r[:i]] = 1 - noisy_unknow_train_data[j, r[:i]]
            noisy_unknow_train_data_tensor = torch.from_numpy(noisy_unknow_train_data).type(torch.FloatTensor)
            noisy_unknow_train_label_tensor = torch.from_numpy(noisy_unknow_train_label).type(torch.LongTensor)

            _, noisy_unknow_train_conf = test(noisy_unknow_train_data_tensor, noisy_unknow_train_label_tensor, net2, criterion, batch_size)
            unknow_score = unknow_score + (np.argmax(noisy_unknow_train_conf, axis = 1)  == noisy_unknow_train_label)

        attack_acc,attack_acc_g,thresh1,attack_acc_c,_ = threshold_based_inference_attack(know_score,know_train_label,ref_score,ref_label,unknow_score,unknow_train_label,test_score,test_label)

        print("Universal threshold: chosen thresh: {:.4f}. attack acc {:.4f}. ".format(thresh1, attack_acc_g))
        print("Class threshold attack acc {:.4f}".format(attack_acc_c))
        best_attack_acc = max(best_attack_acc, attack_acc)
        if best_attack_acc == attack_acc:
            best_flip = i
        print("Current best attack at flip", best_flip, ": ", best_attack_acc)
        sys.stdout.flush()

    print("Best label-only attack at flip", best_flip, ": ", best_attack_acc)

