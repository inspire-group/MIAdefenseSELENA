# -*- coding: utf-8 -*-
import argparse
import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import AverageMeter, accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
small_delta = 1e-30
class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(100,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )          
        self.labels=nn.Sequential(
            nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*2,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':    
                nn.init.normal_(self.state_dict()[key], std=0.01)               
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()

    def forward(self, x1, l):
        out_x1 = self.features(x1)        
        out_l = self.labels(l)            
        is_member =self.combine(torch.cat((out_x1,out_l),1))        
        return self.output(is_member)

class InferenceAttack_HZ2(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_HZ2, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(100,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )
        self.features2=nn.Sequential(
            nn.Linear(100,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )                     
        self.labels=nn.Sequential(
            nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*3,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':    
                nn.init.normal_(self.state_dict()[key], std=0.01)               
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()

    def forward(self, x1, x2, l):
        out_x1 = self.features(x1)
        out_x2 = self.features2(x2)
        out_l = self.labels(l)
        is_member =self.combine(torch.cat((out_x1, out_x2, out_l),1))        
        return self.output(is_member)

def train_attack(infer_data, labels, attack_infer_data, attack_labels, attack_model, attack_criterion, attack_optimizer, batch_size, mode=1, split_ai_infer_data = None, split_ai_attack_infer_data=None):
    # switch to train mode
    attack_model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    len_data = min(len(labels), len(attack_labels))
    len_t = int(np.ceil(len_data/batch_size))
 
    for batch_ind in range(0, len_t):
        end_idx = min((batch_ind+1)*batch_size, len_data)
        outputs = infer_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)        
        outputs_non = attack_infer_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets_attack = attack_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        if mode == 2:
            split_ai_outputs = (split_ai_infer_data[batch_ind*batch_size: end_idx]).to(device, torch.float)
            split_ai_outputs_non = (split_ai_attack_infer_data[batch_ind*batch_size: end_idx]).to(device, torch.float)
            comb_split_ai_inputs = torch.cat((split_ai_outputs, split_ai_outputs_non))

        comb_inputs = torch.cat((outputs,outputs_non))
        comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).to(device, torch.float)
        
        one_hot_tr = torch.zeros(comb_inputs.size()[0],comb_inputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).data, 1)
        if mode == 1: 
            attack_output = attack_model(comb_inputs, target_one_hot).view([-1])
        else:
            attack_output = attack_model(comb_inputs, comb_split_ai_inputs, target_one_hot).view([-1])

        att_labels = torch.zeros((outputs.shape[0]+outputs_non.shape[0]))
        att_labels [:outputs.shape[0]] =1.0
        att_labels [outputs.shape[0]:] =0.0
        is_member_labels = att_labels.to(device, torch.float)
        
        loss_attack = attack_criterion(attack_output, is_member_labels)

        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy()>0.5), (is_member_labels.data.cpu().numpy()> 0.5)))        

        losses.update(loss_attack.data, comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])
        
        # compute gradient and do SGD step
        attack_optimizer.zero_grad()
        loss_attack.backward()
        attack_optimizer.step()

    return (losses.avg, top1.avg)


def test_attack(infer_data, labels, attack_infer_data, attack_labels, attack_model, attack_criterion, batch_size, mode =1, split_ai_infer_data=None, split_ai_attack_infer_data=None):
    attack_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    len_data = min(len(labels), len(attack_labels))
    len_t = int(np.ceil(len_data/batch_size))

    member_prob = np.zeros(len_data)
    nonmember_prob = np.zeros(len_data)

    for batch_ind in range(len_t):
        end_idx = min(len_data, (batch_ind+1)*batch_size)
        outputs = infer_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)
        outputs_non = attack_infer_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets_attack = attack_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        if mode == 2:
            split_ai_outputs = (split_ai_infer_data[batch_ind*batch_size: end_idx]).to(device, torch.float)
            split_ai_outputs_non = (split_ai_attack_infer_data[batch_ind*batch_size: end_idx]).to(device, torch.float)
            comb_split_ai_inputs = torch.cat((split_ai_outputs, split_ai_outputs_non))

        comb_inputs = torch.cat((outputs,outputs_non))
        comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).to(device,torch.float)      
        
        one_hot_tr = torch.zeros(comb_inputs.size()[0],comb_inputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).view([-1,1]).data,1)

        if mode == 1: 
            attack_output = attack_model(comb_inputs, target_one_hot).view([-1])
        else:
            attack_output = attack_model(comb_inputs, comb_split_ai_inputs,target_one_hot).view([-1])

        att_labels = torch.zeros((outputs.shape[0]+outputs_non.shape[0]))
        att_labels [:outputs.shape[0]] =1.0
        att_labels [outputs.shape[0]:] =0.0

        is_member_labels = att_labels.to(device,torch.float)      
        
        loss = attack_criterion(attack_output, is_member_labels)
        
        member_prob[batch_ind*batch_size: end_idx]= attack_output.data.cpu().numpy()[: outputs.shape[0]]
        nonmember_prob[batch_ind*batch_size: end_idx]= attack_output.data.cpu().numpy()[outputs.shape[0]:]

        prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(is_member_labels.data.cpu().numpy()> 0.5)))
        losses.update(loss.data, comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])
    
    return (losses.avg, top1.avg,member_prob,nonmember_prob)

def get_l2_loss(shadow_splitai_predictions, sample_predictions, sample_labels):
    #calculte l2 loss given prediction vectors(N,C) and labels(N), where N is the size of samples and C is the number of class.
    #lower is likely to be samples
    return np.sum((shadow_splitai_predictions - sample_predictions)**2,axis=1)

def get_ce_loss(shadow_splitai_predictions, sample_predictions, sample_labels):
    #calculte ce loss given prediction vectors(N,C) and labels(N), where N is the size of samples and C is the number of class.
    #lower is likely to be samples
    outputs = sample_predictions.copy()
    outputs[outputs<=0] = small_delta
    return np.sum(-shadow_splitai_predictions*np.log(outputs),axis=1)

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


def system_attack(split_ai_infer_train_tr, infer_train_tr, train_label_tr, split_ai_infer_train_te, infer_train_te, train_label_te, split_ai_infer_attack, infer_attack, attack_label, split_ai_infer_test, infer_test, test_label, num_class, attack_epochs=150, batch_size=128*4, seed=123456789):
    #### assume 50% as random guess baseline
    len1, len2 = min(len(train_label_tr), len(attack_label)),min(len(train_label_te), len(test_label))

    split_ai_infer_train_tr, infer_train_tr, train_label_tr = split_ai_infer_train_tr[:len1], infer_train_tr[:len1], train_label_tr[:len1]
    split_ai_infer_attack, infer_attack, attack_label = split_ai_infer_attack[:len1], infer_attack[:len1], attack_label[:len1]
    split_ai_infer_train_te, infer_train_te, train_label_te = split_ai_infer_train_te[:len2], infer_train_te[:len2], train_label_te[:len2]
    split_ai_infer_test, infer_test, test_label = split_ai_infer_test[:len2], infer_test[:len2], test_label[:len2]

    assert(len(infer_train_tr)==len(infer_attack))
    assert(len(infer_train_te)==len(infer_test))

    print("System Attacks:", len(train_label_tr), len(train_label_te), len(attack_label), len(test_label))

    print(train_label_tr[:20])
    print(train_label_te[:20])
    print(test_label[:20])
    print(attack_label[:20])
    print ('classifier acc on attack training set: {:.4f},{:.4f}.\nclassifier acc on attack test set:     {:.4f},{:.4f}.'.format(np.mean(train_label_tr==np.argmax(infer_train_tr,axis=1)), np.mean(attack_label==np.argmax(infer_attack,axis=1)), np.mean(train_label_te==np.argmax(infer_train_te,axis=1)), np.mean(test_label==np.argmax(infer_test,axis=1))))
    print("batch_size: ", batch_size)


    print("CE loss")
    tr_ces = get_ce_loss(split_ai_infer_train_tr, infer_train_tr, train_label_tr)
    attack_ces = get_ce_loss(split_ai_infer_attack, infer_attack, attack_label)
    te_ces = get_ce_loss(split_ai_infer_train_te, infer_train_te, train_label_te)
    test_ces = get_ce_loss(split_ai_infer_test, infer_test, test_label)
    print("Avg CE loss: train attack:{:.4f}/{:.4f}, test attack:{:.4f}/{:.4f}".format(np.mean(tr_ces), np.mean(attack_ces), np.mean(te_ces), np.mean(test_ces)))
    ce_acc1,ce_acc_g1,thresh11,ce_acc_c1,_ = threshold_based_inference_attack(tr_ces,train_label_tr,attack_ces,attack_label,te_ces,train_label_te,test_ces,test_label)
    ce_acc2,ce_acc_g2,thresh13,ce_acc_c2,_ = threshold_based_inference_attack(-tr_ces,train_label_tr,-attack_ces,attack_label,-te_ces,train_label_te,-test_ces,test_label)
    print("Threshold: ", thresh11, thresh13)

    print("L2 Dist")
    tr_l2s = get_l2_loss(split_ai_infer_train_tr, infer_train_tr, train_label_tr)
    attack_l2s = get_l2_loss(split_ai_infer_attack, infer_attack, attack_label)
    te_l2s = get_l2_loss(split_ai_infer_train_te, infer_train_te, train_label_te)
    test_l2s = get_l2_loss(split_ai_infer_test, infer_test, test_label)
    print("Avg L2 Dist: train attack:{:.4f}/{:.4f}, test attack:{:.4f}/{:.4f}".format(np.mean(tr_l2s), np.mean(attack_l2s), np.mean(te_l2s), np.mean(test_l2s)))
    l2_acc1,l2_acc_g1,thresh21,l2_acc_c1,_ = threshold_based_inference_attack(tr_l2s,train_label_tr,attack_l2s,attack_label,te_l2s,train_label_te,test_l2s,test_label)
    l2_acc2,l2_acc_g2,thresh23,l2_acc_c2,_ = threshold_based_inference_attack(-tr_l2s,train_label_tr,-attack_l2s,attack_label,-te_l2s,train_label_te,-test_l2s,test_label)
    print("Threshold ", thresh21, thresh23)

    infer_train_te_tensor = torch.from_numpy(infer_train_te).type(torch.FloatTensor)
    train_label_te_tensor = torch.from_numpy(train_label_te).type(torch.LongTensor)
    split_ai_infer_train_te_tensor = torch.from_numpy(split_ai_infer_train_te).type(torch.FloatTensor)
    infer_test_tensor = torch.from_numpy(infer_test).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)
    split_ai_infer_test_tensor = torch.from_numpy(split_ai_infer_test).type(torch.FloatTensor)

    att_model = InferenceAttack_HZ(num_class).to(device, torch.float)
    att_criterion = nn.MSELoss().to(device, torch.float)
    att_optimizer = optim.Adam(att_model.parameters(),lr=0.0001)

    att_model2 = InferenceAttack_HZ2(num_class).to(device, torch.float)
    att_criterion2 = nn.MSELoss().to(device, torch.float)
    att_optimizer2 = optim.Adam(att_model2.parameters(),lr=0.0001)

    best_attack_acc1 = 0.0
    best_attack_acc2 = 0.0
    np.random.seed(seed)

    for epoch in range(0, attack_epochs):
        r= np.arange(len1)
        np.random.shuffle(r)
        train_label_tr = train_label_tr[r]
        infer_train_tr = infer_train_tr[r]
        split_ai_infer_train_tr = split_ai_infer_train_tr[r]
        r= np.arange(len1)
        np.random.shuffle(r)
        attack_label = attack_label[r]
        infer_attack = infer_attack[r]
        split_ai_infer_attack = split_ai_infer_attack[r]

        infer_train_tr_tensor = torch.from_numpy(infer_train_tr).type(torch.FloatTensor)
        train_label_tr_tensor = torch.from_numpy(train_label_tr).type(torch.LongTensor)
        split_ai_infer_train_tr_tensor = torch.from_numpy(split_ai_infer_train_tr).type(torch.FloatTensor)
        infer_attack_tensor = torch.from_numpy(infer_attack).type(torch.FloatTensor)
        attack_label_tensor = torch.from_numpy(attack_label).type(torch.LongTensor)
        split_ai_infer_attack_tensor = torch.from_numpy(split_ai_infer_attack).type(torch.FloatTensor)

        train_loss, train_attack_acc = train_attack(torch.from_numpy(split_ai_infer_train_tr - infer_train_tr).type(torch.FloatTensor), train_label_tr_tensor,
                                         torch.from_numpy(split_ai_infer_attack - infer_attack).type(torch.FloatTensor), attack_label_tensor, att_model, att_criterion, att_optimizer, batch_size)
        test_loss, test_attack_acc, mem, nonmem = test_attack(torch.from_numpy(split_ai_infer_train_te - infer_train_te).type(torch.FloatTensor), train_label_te_tensor,
                                         torch.from_numpy(split_ai_infer_test -infer_test).type(torch.FloatTensor), test_label_tensor, att_model, att_criterion, batch_size)
        best_attack_acc1 = max(best_attack_acc1, test_attack_acc)


        train_loss, train_attack_acc = train_attack(infer_train_tr_tensor, train_label_tr_tensor,
                                         infer_attack_tensor, attack_label_tensor, att_model2, att_criterion2, att_optimizer2, batch_size, mode = 2, split_ai_infer_data = split_ai_infer_train_tr_tensor, split_ai_attack_infer_data = split_ai_infer_attack_tensor)
        test_loss, test_attack_acc, mem, nonmem = test_attack(infer_train_te_tensor, train_label_te_tensor,
                                         infer_test_tensor, test_label_tensor, att_model2, att_criterion2, batch_size, mode =2, split_ai_infer_data = split_ai_infer_train_te_tensor, split_ai_attack_infer_data = split_ai_infer_test_tensor)
        best_attack_acc2 = max(best_attack_acc2, test_attack_acc)


    print('Model1 Best NN Attack Acc: {:.4f}'.format(best_attack_acc1))
    print('Model2 Best NN Attack Acc: {:.4f}'.format(best_attack_acc2))

    best_attack_acc = max(best_attack_acc1, best_attack_acc2, ce_acc1, ce_acc2, l2_acc_c1, l2_acc2)
    print("\nBEST ATTACK ACC: {:.4f}. \nNN1/NN2: {:.4f}/{:.4f}. \nGlobal|Class: \nLow Threshold \nCE:{:.4f}|{:.4f}. \nL2loss: {:.4f}|{:.4f}. ".format(best_attack_acc, best_attack_acc1, best_attack_acc2, ce_acc_g2, ce_acc_c2, l2_acc_g2, l2_acc_c2))
    print("High Threshold \nCE:{:.4f}|{:.4f}. \nL2loss:{:.4f}|{:.4f}. ".format(ce_acc_g1, ce_acc_c1, l2_acc_g1, l2_acc_c1))
