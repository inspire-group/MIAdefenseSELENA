# -*- coding: utf-8 -*-
import argparse
import os
import random
import sys
import yaml
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
from purchase import PurchaseClassifier
from adaptive_attack import system_attack
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf

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

def selena_test(test_data, test_labels, model, criterion, args):
    model.eval()

    num_class = args.num_class
    batch_size = args.batch_size

    infer_np = np.zeros((test_data.shape[0], num_class))

    len_t =  int(np.ceil(len(test_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)

        features = test_data[batch_ind*batch_size: end_idx]
        labels = test_labels[batch_ind*batch_size: end_idx]

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.long)
        outputs = model(inputs)
        infer_np[batch_ind*batch_size:end_idx] = (F.softmax(outputs, dim=1)).detach().cpu().numpy()

    return infer_np

def main():
    parser = argparse.ArgumentParser(description = 'Setting for Purchase datataset')
    parser.add_argument('--K', type = int, default = 25, help = 'total sub-models in split-ai')
    parser.add_argument('--L', type = int, default = 10, help = 'non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type = int, default =150, help = 'attack epochs in NN attack')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')
    parser.add_argument('--known_ratio', type=float, default=0.5, help='known ratio of member samples by attacker')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    load_name = str(split_model) + '_' + str(non_model)

    DATASET_PATH = os.path.join(root_dir, 'purchase',  'data')
    checkpoint_path = os.path.join(root_dir, 'purchase', 'checkpoints', 'K_L', load_name)
    checkpoint_path_selena = os.path.join(checkpoint_path, 'selena')
    checkpoint_path_shadow = os.path.join(checkpoint_path, 'shadow', str(args.known_ratio))
    print(checkpoint_path_shadow, checkpoint_path_selena)
    

    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'attacker', 'tr_data.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'attacker', 'te_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))
    full_train_data = np.concatenate((train_data_tr_attack, train_data_te_attack), axis = 0)
    full_train_label = np.concatenate((train_label_tr_attack, train_label_te_attack), axis = 0)
    test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'))
    test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))
    ref_data = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'))
    ref_label = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'))
    all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))

    len_train_data = len(full_train_label)
    known_num = int(args.known_ratio*len_train_data)
    unknown_num = len_train_data - known_num
    attack_know_data = full_train_data[:known_num, :]
    attack_know_label = full_train_label[:known_num]
    attack_unknow_data = full_train_data[known_num:, :]
    attack_unknow_label = full_train_label[known_num:]
    
    ##print first 20 labels for each set for sanity check that the corresponding set are consistent with all evalutaions
    print(attack_know_label[:20])
    print(attack_unknow_label[:20])
    print(test_label[:20])
    print(ref_label[:20])
    feature_len = ref_data.shape[1]

    attack_unknow_data = attack_unknow_data[:, :feature_len]
    raw_train_data = attack_know_data.copy()

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    print("Attack Training: # of train data: {:d}, # of ref data: {:d}".format(int(len(attack_know_label)), len(ref_data)))
    print("Attack Testing: # of train data: {:d}, # of test data: {:d}".format(int(len(attack_unknow_label)), len(test_data)))
    criterion = nn.CrossEntropyLoss().to(device, torch.float)
    np.random.seed(0)

    attack_know_data_tensor = torch.from_numpy(attack_know_data).type(torch.FloatTensor)
    attack_know_label_tensor = torch.from_numpy(attack_know_label).type(torch.LongTensor)
    attack_unknow_data_tensor = torch.from_numpy(attack_unknow_data).type(torch.FloatTensor)
    attack_unknow_label_tensor = torch.from_numpy(attack_unknow_label).type(torch.LongTensor)

    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)
    all_test_data_tensor = torch.from_numpy(all_test_data).type(torch.FloatTensor)
    all_test_label_tensor = torch.from_numpy(all_test_label).type(torch.LongTensor)

    net = PurchaseClassifier()
    train_mode = 1
    test_mode = 2
    print("attacker know set")
    infer_attack_know = splitai_test(attack_know_data_tensor, attack_know_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_shadow, train_mode, args, True)
    know_acc, know_conf = print_acc_conf(infer_attack_know, attack_know_label)
    print("attacker unknow set")
    infer_attack_unknow = splitai_test(attack_unknow_data_tensor, attack_unknow_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_shadow, test_mode, args, True)
    unknow_acc, unknow_conf = print_acc_conf(infer_attack_unknow, attack_unknow_label)
    print("test set")
    infer_test_conf = splitai_test(test_data_tensor, test_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_shadow, test_mode, args, True)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference set")
    infer_ref_conf = splitai_test(ref_data_tensor, ref_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_shadow, test_mode, args, True)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    print("For comparison on shadow splitai output")
    print("avg acc on: know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_acc,  unknow_acc,  test_acc,  ref_acc))
    print("avg conf on know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_conf, unknow_conf, test_conf, ref_conf))

    net2 = PurchaseClassifier()
    resume = checkpoint_path_selena +'/model_best.pth.tar'
    print('==> Resuming from checkpoint:'+resume)
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net2.load_state_dict(checkpoint['state_dict'])
    net2 = net2.to(device, torch.float)

    attack_know_data = attack_know_data[:, :feature_len]
    attack_know_data_tensor = torch.from_numpy(attack_know_data).type(torch.FloatTensor)

    print("selena output labels")
    print("attacker know trainset")
    selena_infer_know_conf = selena_test(attack_know_data_tensor, attack_know_label_tensor, net2, criterion, args)
    know_acc, know_conf = print_acc_conf(selena_infer_know_conf, attack_know_label)
    print("attacker unknow trainset")
    selena_infer_unknow_conf = selena_test(attack_unknow_data_tensor, attack_unknow_label_tensor, net2, criterion, args)
    unknow_acc, unknow_conf = print_acc_conf(selena_infer_unknow_conf, attack_unknow_label)
    print("test set")
    selena_infer_test_conf = selena_test(test_data_tensor, test_label_tensor, net2, criterion, args)
    test_acc, test_conf= print_acc_conf(selena_infer_test_conf, test_label)
    print("reference set")
    selena_infer_ref_conf = selena_test(ref_data_tensor, ref_label_tensor, net2, criterion, args)
    ref_acc, ref_conf=print_acc_conf(selena_infer_ref_conf, ref_label)

    print("For comparison on final selena output")
    print("avg acc  on know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_acc,  unknow_acc,  test_acc,  ref_acc))
    print("avg conf on know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_conf, unknow_conf, test_conf, ref_conf))

    system_attack(infer_attack_know, selena_infer_know_conf, attack_know_label, infer_attack_unknow, selena_infer_unknow_conf, attack_unknow_label, infer_ref_conf[:known_num], selena_infer_ref_conf[:known_num], ref_label[:known_num], infer_test_conf[:unknown_num], selena_infer_test_conf[:unknown_num], test_label[:unknown_num], num_class=num_class,attack_epochs=attack_epochs,batch_size=batch_size)

if __name__ == '__main__':
    main()