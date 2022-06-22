# -*- coding: utf-8 -*-
import argparse
import os
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
from dsq_attack import system_attack
from binary_flip_noise_attack import flip_noise_attack
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument('--flip_range', type = int, default = 30, help = 'flip range in label-only attacks')
    parser.add_argument('--nruns', type = int, default = 100, help = 'repeated runs for a fixed flip range')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    flip_range = args.flip_range
    nruns = args.nruns
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

    ##print first 20 labels for each set for sanity check that the corresponding set are consistent with all evalutaions
    print(train_label_tr_attack[:20])
    print(train_label_te_attack[:20])
    print(test_label[:20])
    print(ref_label[:20])
    feature_len = ref_data.shape[1]

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    print("Attack Training: # of train data: {:d}, # of ref data: {:d}".format(int(len(train_data_tr_attack)), len(ref_data)))
    print("Attack Testing: # of train data: {:d}, # of test data: {:d}".format(int(len(train_data_te_attack)), len(test_data)))

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

    net = PurchaseClassifier()
    resume = checkpoint_path_selena +'/model_best.pth.tar'
    print('==> Resuming from checkpoint:'+resume)
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device, torch.float)
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    train_data_tr_attack = train_data_tr_attack[:, :feature_len]
    train_data_te_attack = train_data_te_attack[:, :feature_len]

    train_data_tr_attack_tensor = torch.from_numpy(train_data_tr_attack).type(torch.FloatTensor)
    train_data_te_attack_tensor = torch.from_numpy(train_data_te_attack).type(torch.FloatTensor)

    train_data_tensor = torch.from_numpy(train_data[:, :feature_len]).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    print("selena output labels")
    print("training set")
    selena_infer_train_conf = selena_test(train_data_tensor, train_label_tensor, net, criterion, args)
    train_acc, train_conf = print_acc_conf(selena_infer_train_conf, train_label)
    print("all test set")
    selena_infer_all_test_conf = selena_test(all_test_data_tensor, all_test_label_tensor, net, criterion, args)
    all_test_acc, all_test_conf= print_acc_conf(selena_infer_all_test_conf, all_test_label)
    print("training tr set")
    selena_infer_train_conf_tr = selena_test(train_data_tr_attack_tensor, train_label_tr_attack_tensor, net, criterion, args)
    tr_acc, tr_conf = print_acc_conf(selena_infer_train_conf_tr, train_label_tr_attack)
    print("training te set")
    selena_infer_train_conf_te= selena_test(train_data_te_attack_tensor, train_label_te_attack_tensor, net, criterion, args)
    te_acc, te_conf = print_acc_conf(selena_infer_train_conf_te, train_label_te_attack)
    print("test set")
    selena_infer_test_conf = selena_test(test_data_tensor, test_label_tensor, net, criterion, args)
    test_acc, test_conf= print_acc_conf(selena_infer_test_conf, test_label)
    print("reference set")
    selena_infer_ref_conf = selena_test(ref_data_tensor, ref_label_tensor, net, criterion, args)
    ref_acc, ref_conf=print_acc_conf(selena_infer_ref_conf, ref_label)

    print("For comparison on final selena output")
    print("avg acc  on train/all test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_acc, all_test_acc ,tr_acc, te_acc, test_acc, ref_acc))
    print("avg conf on train/all_test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_conf, all_test_conf, tr_conf, te_conf, test_conf, ref_conf))

    system_attack(selena_infer_train_conf_tr, train_label_tr_attack, selena_infer_train_conf_te, train_label_te_attack, selena_infer_ref_conf, ref_label, selena_infer_test_conf, test_label, num_class=num_class,attack_epochs=attack_epochs,batch_size=batch_size)

    flip_noise_attack(net, train_data_tr_attack, train_label_tr_attack, train_data_te_attack, train_label_te_attack, ref_data, ref_label, test_data, test_label, num_class, batch_size, flip_range=flip_range,repeated=nruns)


if __name__ == '__main__':
    main()