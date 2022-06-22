# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import sys
import yaml

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

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

def main():
    parser = argparse.ArgumentParser(description='undefend training for Purchase dataset')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    attack_epochs = args.attack_epochs

    DATASET_PATH = os.path.join(root_dir, 'purchase',  'data')
    checkpoint_path = os.path.join(root_dir, 'purchase', 'checkpoints', 'undefend')
    print(checkpoint_path)    
    
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))
    test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))
    ref_label = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'))

    #print first 20 labels for each subset, for checking with other experiments
    print(train_label_tr_attack[:20])
    print(train_label_te_attack[:20])
    print(test_label[:20])
    print(ref_label[:20])

    infer_train_conf_tr = np.load(os.path.join(DATASET_PATH, 'memguard', 'defense_results', 'memguard_tr.npy'))
    infer_train_conf_te = np.load(os.path.join(DATASET_PATH, 'memguard', 'defense_results', 'memguard_te.npy'))
    infer_ref_conf = np.load(os.path.join(DATASET_PATH, 'memguard', 'defense_results', 'memguard_ref.npy'))
    infer_test_conf = np.load(os.path.join(DATASET_PATH, 'memguard', 'defense_results', 'memguard_test.npy'))

    system_attack(infer_train_conf_tr, train_label_tr_attack, infer_train_conf_te, train_label_te_attack, infer_ref_conf, ref_label, infer_test_conf, test_label, num_class=num_class, attack_epochs=attack_epochs,batch_size=batch_size)

if __name__ == '__main__':
    main()