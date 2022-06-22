import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets

from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from resnet import resnet18
from cifar_utils import transform_test, Cifardata

def cw_l2_attack(model, o_images, arc_images, images, labels, device, targeted=False, c=1e-2, kappa=0, max_iter=1000, learning_rate=1e-2):

    images = images.to(device, torch.float)
    labels = labels.to(device, torch.long)
    o_images = o_images.to(device, torch.float)
    arc_images = arc_images.to(device, torch.float)

    attack_images = (torch.zeros((len(labels), 3, 32, 32))).to(device, torch.float)

    best_labels = np.ones(len(labels)) * (-1)
    record_dists = np.ones(len(labels)) * np.inf

    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs-one_hot_labels*10000, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)    
    optimizer = optim.Adam([w], lr=learning_rate)
    
    a = 1/2*(nn.Tanh()(w+arc_images) + 1)

    prev = 1e10

    for step in range(max_iter):
        loss1 = nn.MSELoss(reduction='sum')(a, images)

        loss2 = 0

        part2 = f(a)
        for ind in range(len(labels)):
            loss2 = loss2 + c[ind] * part2[ind]

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        a = 1/2*(nn.Tanh()(w+arc_images) + 1)

        noises  = torch.sqrt(torch.sum((o_images-a)**2, dim = [1, 2, 3]))
        pred = (torch.argmax(model(a), dim=1)).detach().cpu().numpy()


        for ind in range(len(labels)):
            if pred[ind] != labels[ind] and noises[ind].item() < record_dists[ind]:
                attack_images[ind] = a[ind]
                best_labels[ind] = pred[ind]
                record_dists[ind] = noises[ind]

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                break
            prev = cost

    return attack_images.detach().cpu().numpy(), best_labels, record_dists

def get_optimal_perturb(model, data, label, infer_conf, batch_size, num_worker, device):

    data_proc = data[label == np.argmax(infer_conf, axis = 1)]
    label_proc = label[label == np.argmax(infer_conf, axis = 1)]

    imageset = Cifardata(data_proc, label_proc, transform_test)
    imageloader = torch.utils.data.DataLoader(imageset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    all_images = np.zeros((len(label), 3, 32, 32))
    all_bestlabel = (-1) * np.ones(len(label))
    all_dists = np.inf*np.ones(len(label))

    all_dists[len(data_proc):] = 0

    repeat_binary_search = 10
    upper_bound  = 1e10
    initial_current = 1e-2

    for batch_ind, (images, labels) in enumerate(imageloader):

        saved_images = np.zeros((len(labels), 3, 32, 32))

        o_bestlabel = (-1) * np.ones(len(labels))
        o_dists = np.inf*np.ones(len(labels))

        base_ind = batch_size*batch_ind
        orig_images = torch.from_numpy((data_proc[base_ind: base_ind + len(labels)])/255.0).type(torch.FloatTensor)
        arctanh_images = torch.from_numpy(np.arctanh(0.999999*(2*(data_proc[base_ind:  base_ind+len(labels)]/255.0)-1))).type(torch.FloatTensor)
        
        upper_bound_list = np.ones(len(labels))*upper_bound
        lower_bound_list = np.zeros(len(labels))
        
        current = np.ones(len(labels))*initial_current
        
        for step in range(repeat_binary_search):
            print("Step: {:d}. ".format(step))
            if repeat_binary_search > 10 and step == repeat_binary_search - 1:
                current = np.ones(len(labels)) * upper_bound
            
            tmp_images, bestlabels, bestl2 = cw_l2_attack(model, orig_images, arctanh_images, images, labels, device, targeted=False, c=current)
            labels = labels.to(device)

            for ind in range(len(labels)):
                if bestlabels[ind] != labels[ind].item() and bestlabels[ind] != -1:
                    o_bestlabel[ind] = bestlabels[ind]
                    o_dists[ind] = bestl2[ind]
                    saved_images[ind] = tmp_images[ind]

                    upper_bound_list[ind] = min(upper_bound_list[ind], current[ind])
                    current[ind] = (lower_bound_list[ind] + upper_bound_list[ind])/2
                else:
                    lower_bound_list[ind] = max(lower_bound_list[ind], current[ind])
                    if upper_bound_list[ind] < 1e9:
                        current[ind] = (lower_bound_list[ind] + upper_bound_list[ind])/2
                    else:
                        current[ind] = current[ind] *10
        for ind in range(len(labels)):
            if o_bestlabel[ind] != labels[ind].item() and o_bestlabel[ind] != -1:
                all_images[base_ind+ind] = saved_images[ind]
                all_bestlabel[base_ind + ind] = o_bestlabel[ind]
                all_dists[base_ind + ind] = o_dists[ind]


    return_dists = []
    return_labels = []
    cnt = 0
    for ind in range(len(label_proc)): 
        if all_bestlabel[ind] !=-1 and all_bestlabel[ind]!= label_proc[ind]:
            cnt = cnt+1
            return_dists.append(all_dists[ind])
            return_labels.append(label_proc[ind])
    
    return_dists.extend(all_dists[len(label_proc):])
    return_labels.extend(label[label != np.argmax(infer_conf, axis = 1)])

    print("find adversarial examples for {:.2f} of data. correct data {:.2f} of all data".format(cnt/len(label_proc), 1.0*len(label_proc)/len(label)))

    return np.array(return_dists), np.array(return_labels)

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

def cw_attack(net, save_path, know_train_data, know_train_label, know_train_conf, unknow_train_data, unknow_train_label, unknow_train_conf, ref_data, ref_label, infer_ref_conf, test_data, test_label, infer_test_conf, device, num_worker, num_class=100,batch_size=256):
    net.eval()

    ####Assume len(know_train_conf)==len(unknow_train_conf), len(ref_label)==len(ttest)label
    total_data = min(len(know_train_conf), len(ref_label))
    
    know_train_data, know_train_label, know_train_conf = know_train_data[:total_data], know_train_label[:total_data], know_train_conf[:total_data]
    ref_data, ref_label, infer_ref_conf = ref_data[:total_data], ref_label[:total_data], infer_ref_conf[:total_data]
    unknow_train_data, unknow_train_label, unknow_train_conf = unknow_train_data[:total_data], unknow_train_label[:total_data], unknow_train_conf[:total_data]
    test_data, test_label, infer_test_conf = test_data[:total_data], test_label[:total_data], infer_test_conf[:total_data]
   
    know_train_perturb, new_know_train_label = get_optimal_perturb(net, know_train_data, know_train_label, know_train_conf, batch_size, num_worker, device)
    ### save the files
    np.save(os.path.join(save_path,"original_know_train_dist.npy"), know_train_perturb)
    np.save(os.path.join(save_path,"new_know_train_label.npy"), new_know_train_label)

    unknow_train_perturb, new_unknow_train_label = get_optimal_perturb(net, unknow_train_data, unknow_train_label, unknow_train_conf, batch_size, num_worker, device)
    ### save the files
    np.save(os.path.join(save_path,"original_unknow_train_dist.npy"), unknow_train_perturb)
    np.save(os.path.join(save_path,"new_unknow_train_label.npy"), new_unknow_train_label)

    test_perturb, new_test_label = get_optimal_perturb(net, test_data, test_label, infer_test_conf, batch_size, num_worker, device)
    ### save the files
    np.save(os.path.join(save_path,"original_test_dist.npy"), test_perturb)
    np.save(os.path.join(save_path,"new_test_label.npy"), new_test_label)

    ref_perturb, new_ref_label = get_optimal_perturb(net, ref_data, ref_label, infer_ref_conf, batch_size, num_worker, device)
    ### save the files
    np.save(os.path.join(save_path,"original_ref_dist.npy"), ref_perturb)
    np.save(os.path.join(save_path,"new_ref_label.npy"), new_ref_label)
    
    print("{:d}/{:d}".format(min(len(know_train_perturb), len(ref_perturb)), min(len(unknow_train_perturb), len(test_perturb))))

    attack_acc,attack_acc_g,thresh1,attack_acc_c,_ = threshold_based_inference_attack(know_train_perturb,know_train_label,ref_perturb,ref_label,unknow_train_perturb,unknow_train_label,test_perturb,test_label)
    print("CW attack: {:.4f}".format(attack_acc))
    print("Universal threshold: chosen thresh: {:.4f}. attack acc {:.4f}. ".format(thresh1, attack_acc_g))
    print("Class threshold attack acc {:.4f}".format(attack_acc_c))