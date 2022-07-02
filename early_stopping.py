# -*- coding: utf-8 -*-
import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
import matplotlib.font_manager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

config_file = './env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'attack'))
sys.path.append(os.path.join(src_dir, 'models'))

from dsq_attack import system_attack
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf, binarydata
from purchase import PurchaseClassifier
from texas import TexasClassifier
from resnet import resnet18
from cifar_utils import Cifardata, ModelwNorm, transform_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def undefendtest(testloader, model, criterion, len_data, args):
    # switch to evaluate mode
    model.eval()

    num_class = args.num_class
    batch_size = args.batch_size

    losses = AverageMeter()
    infer_np = np.zeros((len_data, num_class))

    for batch_ind, (inputs, targets) in enumerate(testloader):
        # compute output
        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        infer_np[batch_ind*batch_size: batch_ind*batch_size+inputs.shape[0]] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()

        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, infer_np)#, logits_np)


def plot(name, xx, yy1, yy2, indice, indicey, acc, attack_acc, y_range):
    ##### name, indice,, indicey, yranges are only for better plotting
    ##### xx, range of epochs
    ##### yy1 classification acc on test set
    ##### yy2 MIA attack accuracy for epoch
    font = {'weight': 'normal',
        'size': 12}
    plt.figure(figsize=[8,6.3])
    plt.rc('font', **font)
    mpl.rcParams['pdf.fonttype'] = 42

    ll = min(len(xx), len(yy1), len(yy2))
    plt.plot(xx[:ll], yy1[:ll], label ='ER test accuracy', linewidth=2, marker = '*')
    plt.plot(xx[:ll], yy2[:ll], label = 'ER MIA accuracy', linewidth=2, marker = 'D')
    plt.xlabel("Epoch", fontsize = 20)

    plt.ylim(0, y_range)
    plt.xlim(1, np.max(xx)+1)

    plt.axhline(y=acc, color = 'purple', linestyle='-.', linewidth=2, label = 'SELENA test accuracy')
    plt.axhline(y=attack_acc, color = 'darkgreen', linestyle='-.',linewidth=2 , label = 'SELENA MIA accuracy')

    plt.text(ll/10, acc-0.01, acc, ha='center', va= 'bottom',fontsize=28)
    plt.text(int(ll*0.9), attack_acc, attack_acc, ha='center', va= 'bottom',fontsize=28)

    plt.legend()
    x_major_locator=MultipleLocator(indice)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    y_major_locator=MultipleLocator(indicey)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=28) 
    plt.savefig(name + ".pdf", dpi = 600, format = 'pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='undefend training for Purchase dataset')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--dataset', type = str, default ='purchase', choices=['purchase','texas','cifar100'], help = 'dataset name')
    parser.add_argument('--num_worker', type=int, default=1, help='number workers')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    if args.dataset == 'purchase':
        args.run_epochs = 30
        args.batch_size = 512
        args.num_class = 100
        net2 = PurchaseClassifier()
    elif args.dataset == 'texas':
        args.run_epochs = 20
        args.batch_size = 128
        args.num_class = 100
        net2 = TexasClassifier()
    else:
        args.run_epochs = 200
        args.batch_size = 256
        args.num_class = 100
        net2_1 = resnet18()
        net2 = ModelwNorm(net2_1)

    net2 = net2.to(device, torch.float)

    batch_size = args.batch_size
    num_class = args.num_class
    attack_epochs = args.attack_epochs
    num_worker = args.num_worker


    DATASET_PATH = os.path.join(root_dir, args.dataset,  'data')
    checkpoint_path = os.path.join(root_dir, args.dataset, 'checkpoints', 'undefend')
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

    if args.dataset == 'cifar100':
        testset = Cifardata(test_data, test_label, transform_test)
        refset = Cifardata(ref_data, ref_label, transform_test)

        trset = Cifardata(train_data_tr_attack, train_label_tr_attack, transform_test)
        teset = Cifardata(train_data_te_attack, train_label_te_attack, transform_test)
    else:

        testset = binarydata(test_data, test_label)
        refset = binarydata(ref_data, ref_label)

        trset = binarydata(train_data_tr_attack, train_label_tr_attack)
        teset = binarydata(train_data_te_attack, train_label_te_attack)

    trloader = torch.utils.data.DataLoader(trset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    
    epoch_dsq_attacks = []
    test_accs = []

    for i in range(1, args.run_epochs+1):
        resume = checkpoint_path +'/Depoch'+str(i)+'.pth.tar'
        print('==> Resuming from checkpoint:'+resume)
        assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resume)
        net2.load_state_dict(checkpoint['state_dict'])
    
        #print("tr set")
        tr_loss, infer_train_conf_tr = undefendtest(trloader, net2, criterion, len(trset), args)
        #tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack)
        #print("te set")
        te_loss, infer_train_conf_te = undefendtest(teloader, net2, criterion, len(teset), args)
        #te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack)
        #print("test set")
        test_loss, infer_test_conf = undefendtest(testloader, net2, criterion, len(testset), args)
        #test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
        #print("reference set")
        ref_loss, infer_ref_conf = undefendtest(refloader, net2, criterion, len(refset), args)
        #ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

        test_acc = np.mean(test_label==np.argmax(infer_test_conf,axis=1))
        attack_acc = system_attack(infer_train_conf_tr, train_label_tr_attack, infer_train_conf_te, train_label_te_attack, infer_ref_conf, ref_label, infer_test_conf, test_label, num_class=num_class, attack_epochs=attack_epochs,batch_size=batch_size)
        epoch_dsq_attacks.append(attack_acc)
        test_accs.append(test_acc)

    if args.dataset == 'purchase':
        ###0.793, 0.543 are from accuracy reported in the paper.
        plot('purchase', np.arange(args.run_epochs), test_accs, epoch_dsq_attacks, 5, 0.2, 0.793, 0.543, 0.9)
    elif args.dataset == 'texas':
        ###0.526, 0.551 are from accuracy reported in the paper.
        plot('texas', np.arange(args.run_epochs), test_accs, epoch_dsq_attacks, 4, 0.2, 0.526, 0.551, 0.85)
    else:
        ###0.746, 0.583 are from accuracy reported in the paper.
        plot('cifar100', np.arange(args.run_epochs), test_accs, epoch_dsq_attacks, 40, 0.2, 0.746, 0.583, 0.9)



if __name__ == '__main__':
    main()