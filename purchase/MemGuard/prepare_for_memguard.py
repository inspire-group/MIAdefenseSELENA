# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import sys
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

config_file = './../../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'attack'))
sys.path.append(os.path.join(src_dir, 'models'))
from dsq_attack import system_attack
from binary_flip_noise_attack import flip_noise_attack
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from purchase import PurchaseClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def undefendtest(test_data, labels, model, criterion, batch_size):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    len_t =  int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))
    logits_np = np.zeros((len(test_data),100))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
        logits_np[batch_ind*batch_size: end_idx] = (outputs).detach().cpu().numpy()
    
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, infer_np, logits_np)

def main():
    parser = argparse.ArgumentParser(description='undefend training for Purchase dataset')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class

    DATASET_PATH = os.path.join(root_dir, 'purchase',  'data')
    checkpoint_path = os.path.join(root_dir, 'purchase', 'checkpoints', 'undefend')
    print(checkpoint_path)    
    
    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))
    train_data = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data.npy'))
    train_label = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label.npy'))
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

    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    train_data_tr_attack_tensor = torch.from_numpy(train_data_tr_attack).type(torch.FloatTensor)
    train_label_tr_attack_tensor = torch.from_numpy(train_label_tr_attack).type(torch.LongTensor)
    train_data_te_attack_tensor = torch.from_numpy(train_data_te_attack).type(torch.FloatTensor)
    train_label_te_attack_tensor = torch.from_numpy(train_label_te_attack).type(torch.LongTensor)
    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)
    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)    

    net2 = PurchaseClassifier()
    resume = checkpoint_path +'/model_best.pth.tar'
    print('==> Resuming from checkpoint:'+resume)
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net2.load_state_dict(checkpoint['state_dict'])
    net2 = net2.to(device, torch.float)


    print("training tr set")
    tr_loss, infer_train_conf_tr, train_logits_tr = undefendtest(train_data_tr_attack_tensor, train_label_tr_attack_tensor, net2, criterion, batch_size)
    tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack)
    print("training te set")
    te_loss, infer_train_conf_te, train_logits_te = undefendtest(train_data_te_attack_tensor, train_label_te_attack_tensor, net2, criterion, batch_size)
    te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack)
    print("test set")
    test_loss, infer_test_conf, test_logits = undefendtest(test_data_tensor, test_label_tensor, net2, criterion, batch_size)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("ref set")
    ref_loss, infer_ref_conf, ref_logits = undefendtest(ref_data_tensor, ref_label_tensor, net2, criterion, batch_size)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    len1 = min(len(train_label_tr_attack),len(ref_label))
    len2 = min(len(train_label_te_attack),len(test_label))
    infer_train_conf_tr, train_logits_tr = infer_train_conf_tr[:len1], train_logits_tr[:len1]
    infer_ref_conf, ref_logits = infer_ref_conf[:len1], ref_logits[:len1]
    infer_train_conf_te, train_logits_te = infer_train_conf_te[:len2], train_logits_te[:len2]
    infer_test_conf, test_logits = infer_test_conf[:len2], test_logits[:len2]

    if not os.path.exists(os.path.join(DATASET_PATH, "memguard", "prediction")):
        mkdir_p(os.path.join(DATASET_PATH, "memguard", "prediction"))
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "infer_train_conf_tr.npy"), infer_train_conf_tr)
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "train_logits_tr.npy"), train_logits_tr)
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "infer_train_conf_te.npy"), infer_train_conf_te)
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "train_logits_te.npy"), train_logits_te)
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "infer_test_conf.npy"), infer_test_conf)
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "test_logits.npy"), test_logits)
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "infer_ref_conf.npy"), infer_ref_conf)
    np.save(os.path.join(DATASET_PATH, "memguard", "prediction", "ref_logits.npy"), ref_logits)

if __name__ == '__main__':
    main()