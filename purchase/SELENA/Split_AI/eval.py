# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import random
import sys
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

def main():
    parser = argparse.ArgumentParser(description = 'Setting for Purchase datataset')
    parser.add_argument('--K', type = int, default = 25, help = 'total sub-models in split-ai')
    parser.add_argument('--L', type = int, default = 10, help = 'non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    load_name = str(split_model) + '_' + str(non_model)

    path_dir = os.path.join(root_dir, 'purchase')
    
    checkpoint_path = os.path.join(path_dir, 'checkpoints', 'K_L', load_name)
    checkpoint_path_splitai = os.path.join(checkpoint_path, 'split_ai')
    checkpoint_path_selena = os.path.join(checkpoint_path, 'selena')
    print(checkpoint_path, checkpoint_path_selena)
    
    DATASET_PATH = os.path.join(path_dir, 'data')
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

    test_data, test_label = test_data[:9866], test_label[:9866]
    ref_data, ref_label = ref_data[:9866], ref_label[:9866]
    all_test_data, all_test_label = np.concatenate((test_data, ref_data), axis=0), np.concatenate((test_label, ref_label), axis=0)
    #all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    #all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))

    #print first 20 labels for each subset, for checking with other experiments
    print(train_label_tr_attack[:20])
    print(train_label_te_attack[:20])
    print(test_label[:20])
    print(ref_label[:20])

    feature_len = test_data.shape[1]

    raw_train_data = train_data.copy()

    train_mode = 1
    test_mode = 2

    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    net = PurchaseClassifier()

    print("Attack Training: # of train data: {:d}, # of ref data: {:d}".format(int(0.5*len(train_data)), len(ref_data)))
    print("Attack Testing: # of train data: {:d}, # of test data: {:d}".format(int(0.5*len(train_data)), len(test_data)))

    train_data_tr_attack_tensor = torch.from_numpy(train_data_tr_attack).type(torch.FloatTensor)
    train_label_tr_attack_tensor = torch.from_numpy(train_label_tr_attack).type(torch.LongTensor)
    train_data_te_attack_tensor = torch.from_numpy(train_data_te_attack).type(torch.FloatTensor)
    train_label_te_attack_tensor = torch.from_numpy(train_label_te_attack).type(torch.LongTensor)

    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)
    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)
    all_test_data_tensor = torch.from_numpy(all_test_data).type(torch.FloatTensor)
    all_test_label_tensor = torch.from_numpy(all_test_label).type(torch.LongTensor)

    print("training set")
    infer_train_conf = splitai_test(train_data_tensor, train_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, train_mode, args, True)
    train_acc, train_conf = print_acc_conf(infer_train_conf, train_label)
    print("all test set")
    infer_all_test_conf = splitai_test(all_test_data_tensor, all_test_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, test_mode, args, True)
    all_test_acc, all_test_conf = print_acc_conf(infer_all_test_conf, all_test_label)
    print("training tr set")
    infer_train_conf_tr = splitai_test(train_data_tr_attack_tensor, train_label_tr_attack_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, train_mode, args, True)
    tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack)
    print("training te set")
    infer_train_conf_te = splitai_test(train_data_te_attack_tensor, train_label_te_attack_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, train_mode, args, True)
    te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack)
    print("test set")
    infer_test_conf = splitai_test(test_data_tensor, test_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, test_mode, args, True)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference set")
    infer_ref_conf = splitai_test(ref_data_tensor, ref_label_tensor, net, criterion, feature_len, raw_train_data, checkpoint_path_splitai, test_mode, args, True)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    print("For comparison on splitai output")
    print("avg acc on: train/all test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_acc, all_test_acc ,tr_acc, te_acc, test_acc, ref_acc))
    print("avg conf on train/all_test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_conf, all_test_conf, tr_conf, te_conf, test_conf, ref_conf))

    system_attack(infer_train_conf_tr, train_label_tr_attack, infer_train_conf_te, train_label_te_attack, infer_ref_conf, ref_label, infer_test_conf, test_label, num_class=num_class,attack_epochs=attack_epochs,batch_size=batch_size)

if __name__ == '__main__':
    main()
