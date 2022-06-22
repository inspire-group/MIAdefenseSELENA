import os
from collections import defaultdict
import sys
import numpy as np
from PIL import Image
import yaml

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from resnet import resnet18
from cifar_utils import transform_test

class Cifarattack(data.Dataset):
    def __init__(self, data, labels, transform, is_flipped, padding_size, hidx, widx):
        self.data = data
        self.transform = transform
        self.labels = labels
        self.is_flipped = is_flipped
        self.padding_size = padding_size
        self.hidx = hidx
        self.widx = widx


    def __getitem__(self, index):
        new_img = np.zeros((3, 32+2*self.padding_size, 32+2*self.padding_size))
        new_img[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size] = self.data[index]
        crop_img = new_img[:, self.hidx: self.hidx+32, self.widx: self.widx+32]
        if self.is_flipped:
            crop_img = np.flip(crop_img, axis = 2)
        label = self.labels[index]
        img =  Image.fromarray((crop_img.transpose(1,2,0).astype(np.uint8)))
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

def test(testloader, model, criterion, batch_size, len_data, num_class, device):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    infer_np = np.zeros((len_data, num_class))

    for batch_ind, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        infer_np[batch_ind*batch_size: batch_ind*batch_size+inputs.shape[0]] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()

        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, infer_np)

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

def aug_attack(net, know_train_data, know_train_label, unknow_train_data, unknow_train_label, ref_data, ref_label, test_data, test_label, device, num_worker, num_class=100,batch_size=256):
    net.eval()
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    len1, len2 = min(len(know_train_label), len(ref_label)),min(len(unknow_train_label), len(test_label))

    know_train_data, know_train_label = know_train_data[:len1], know_train_label[:len1]
    ref_data, ref_label = ref_data[:len1], ref_label[:len1]
    unknow_train_data, unknow_train_label = unknow_train_data[:len2], unknow_train_label[:len2]
    test_data, test_label = test_data[:len2], test_label[:len2]

    know_train_record = np.zeros((len1, 81*2))
    unknow_train_record = np.zeros((len2, 81*2))
    ref_record = np.zeros((len1, 81*2))
    test_record = np.zeros((len2, 81*2))

    for i in range(9):
        for j in range(9):
            t_knowtrainset = Cifarattack(know_train_data[:len1], know_train_label[:len1], transform_test, False, 4, i, j)
            t_unknowtrainset = Cifarattack(unknow_train_data[:len2], unknow_train_label[:len2], transform_test, False, 4, i, j)        
            t_testset = Cifarattack(test_data[:len2], test_label[:len2], transform_test, False, 4, i, j)
            t_refset = Cifarattack(ref_data[:len1], ref_label[:len1], transform_test, False, 4, i, j)
            t_knowtrainloader = torch.utils.data.DataLoader(t_knowtrainset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            t_unknowtrainloader = torch.utils.data.DataLoader(t_unknowtrainset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            t_testloader = torch.utils.data.DataLoader(t_testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            t_refloader = torch.utils.data.DataLoader(t_refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            _, infer_know_train_conf = test(t_knowtrainloader, net, criterion, batch_size, len(t_knowtrainset), num_class, device)
            _, infer_unknow_train_conf = test(t_unknowtrainloader, net, criterion, batch_size, len(t_unknowtrainset), num_class, device)
            _, infer_test_conf = test(t_testloader, net, criterion, batch_size, len(t_testset), num_class, device)
            _, infer_ref_conf = test(t_refloader, net, criterion, batch_size, len(t_refset), num_class, device)
            know_train_record[:, 9*i+j] = (np.argmax(infer_know_train_conf, axis=1)==know_train_label[:len1])
            unknow_train_record[:, 9*i+j] = (np.argmax(infer_unknow_train_conf, axis=1)==unknow_train_label[:len2])
            test_record[:, 9*i+j] = (np.argmax(infer_test_conf, axis=1)==test_label[:len2])
            ref_record[:, 9*i+j] = (np.argmax(infer_ref_conf, axis=1)==ref_label[:len1])

            t_knowtrainset = Cifarattack(know_train_data[:len1], know_train_label[:len1], transform_test, True, 4, i, j)
            t_unknowtrainset = Cifarattack(unknow_train_data[:len2], unknow_train_label[:len2], transform_test, True, 4, i, j)         
            t_testset = Cifarattack(test_data[:len2], test_label[:len2], transform_test, True, 4, i, j)
            t_refset = Cifarattack(ref_data[:len1], ref_label[:len1], transform_test, True, 4, i, j)
            t_knowtrainloader = torch.utils.data.DataLoader(t_knowtrainset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            t_unknowtrainloader = torch.utils.data.DataLoader(t_unknowtrainset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            t_testloader = torch.utils.data.DataLoader(t_testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            t_refloader = torch.utils.data.DataLoader(t_refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            _, infer_know_train_conf = test(t_knowtrainloader, net, criterion, batch_size, len(t_knowtrainset), num_class, device)
            _, infer_unknow_train_conf = test(t_unknowtrainloader, net, criterion, batch_size, len(t_unknowtrainset), num_class, device)
            _, infer_test_conf = test(t_testloader, net, criterion, batch_size, len(t_testset), num_class, device)
            _, infer_ref_conf = test(t_refloader, net, criterion, batch_size, len(t_refset), num_class, device)
            _, infer_ref_conf = test(t_refloader, net, criterion, batch_size, len(t_refset), num_class, device)
            know_train_record[:, 9*i+j+81] = (np.argmax(infer_know_train_conf, axis=1)==know_train_label[:len1])
            unknow_train_record[:, 9*i+j+81] = (np.argmax(infer_unknow_train_conf, axis=1)==unknow_train_label[:len2])
            test_record[:, 9*i+j+81] = (np.argmax(infer_test_conf, axis=1)==test_label[:len2])
            ref_record[:, 9*i+j+81] = (np.argmax(infer_ref_conf, axis=1)==ref_label[:len2])

    know_train_score = np.mean(know_train_record, axis = 1)
    unknow_train_score = np.mean(unknow_train_record, axis = 1)
    test_score = np.mean(test_record, axis = 1)
    ref_score = np.mean(ref_record, axis = 1)

    attack_acc,attack_acc_g,thresh1,attack_acc_c,_ = threshold_based_inference_attack(know_train_score,know_train_label,ref_score,ref_label,unknow_train_score, unknow_train_label,test_score,test_label)
    print("Augmentation attack: {:.4f}".format(attack_acc))
    print("Universal threshold: chosen thresh: {:.4f}. attack acc {:.4f}. ".format(thresh1, attack_acc_g))
    print("Class threshold attack acc {:.4f}".format(attack_acc_c))