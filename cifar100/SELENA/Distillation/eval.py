import argparse
import os
import sys
import shutil
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

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
from resnet import resnet18
from cifar_utils import transform_train, transform_test, Cifardata, DistillCifardata, WarmUpLR, ModelwNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def selena_test(testloader, model, criterion, len_data, args):
    model.eval()

    batch_size = args.batch_size
    num_class = args.num_class

    losses = AverageMeter()
    infer_np = np.zeros((len_data, num_class))

    for batch_ind, (features, labels) in enumerate(testloader):

        inputs = features.to(device, torch.float)
        targets = labels.to(device, torch.long)
        outputs = model(inputs)

        infer_np[batch_ind*batch_size:batch_ind*batch_size + inputs.shape[0]] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()

        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, infer_np)

def main():
    parser = argparse.ArgumentParser(description='setting for cifar100')
    parser.add_argument('--K', type=int, default=25, help='total sub-models in split-ai')
    parser.add_argument('--L', type=int, default=10, help='non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')
    parser.add_argument('--classifier_epochs', type=int, default=200, help='classifier epochs in distillation')
    parser.add_argument('--print_epoch_splitai', type=int, default=5, help='print splitai single model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--warmup', type=int, default=1, help='warm up epochs')
    parser.add_argument('--num_worker', type=int, default=1, help='number workers')
    parser.add_argument('--num_class', type=int, default=100, help='num class')


    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    classifer_epochs = args.classifier_epochs
    print_epoch_splitai = args.print_epoch_splitai
    load_name = str(split_model) + '_' + str(non_model)
    warmup = args.warmup
    num_worker = args.num_worker

    DATASET_PATH = os.path.join(root_dir, 'cifar100',  'data')
    checkpoint_path = os.path.join(root_dir, 'cifar100', 'checkpoints', 'K_L', load_name)
    checkpoint_path_splitai = os.path.join(checkpoint_path, 'split_ai')
    checkpoint_path_selena =os.path.join(checkpoint_path, 'selena')
    print(checkpoint_path, checkpoint_path_selena)

    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'defender', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'defender', 'te_label.npy'))
    train_data = np.concatenate((train_data_tr_attack, train_data_te_attack), axis = 0)
    train_label = np.concatenate((train_label_tr_attack, train_label_te_attack), axis = 0)
    test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'))
    test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))
    ref_data = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'))
    ref_label = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'))
    all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))

    #print first 20 labels for each subset, for checking with other experiments
    print(train_label_tr_attack[:20, 0])
    print(train_label_te_attack[:20, 0])
    print(test_label[:20])
    print(ref_label[:20])

    testset = Cifardata(test_data, test_label, transform_test)
    refset = Cifardata(ref_data, ref_label, transform_test)
    alltestset = Cifardata(all_test_data, all_test_label, transform_test)

    criterion = (nn.CrossEntropyLoss()).to(device, torch.float)
    net2_t = resnet18()
    net2 = ModelwNorm(net2_t)
    net2 = net2.to(device, torch.float)

    resume = checkpoint_path_selena +'/model_best.pth.tar'
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net2.load_state_dict(checkpoint['state_dict'])
    net2 = net2.to(device, torch.float)

    train_label_tr_attack = train_label_tr_attack[:, 0]
    train_label_te_attack = train_label_te_attack[:, 0]
    trset = Cifardata(train_data_tr_attack, train_label_tr_attack, transform_test)
    trloader = torch.utils.data.DataLoader(trset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    teset = Cifardata(train_data_te_attack, train_label_te_attack, transform_test)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    train_label = train_label[:, 0]
    traintestset = Cifardata(train_data, train_label, transform_test)
    traintestloader = torch.utils.data.DataLoader(traintestset, batch_size=batch_size, shuffle=False, num_workers=num_worker)


    alltestloader = torch.utils.data.DataLoader(alltestset, batch_size=batch_size, shuffle = False, num_workers = num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)


    print("selena output labels")
    print("training set")
    train_loss, selena_infer_train_conf = selena_test(traintestloader, net2, criterion, len(traintestset), args)
    train_acc, train_conf = print_acc_conf(selena_infer_train_conf, train_label)
    print("all test set")
    all_test_loss, selena_infer_all_test_conf = selena_test(alltestloader, net2, criterion, len(alltestset), args)
    all_test_acc, all_test_conf = print_acc_conf(selena_infer_all_test_conf, all_test_label)
    print("tr set")
    tr_loss, selena_infer_train_conf_tr = selena_test(trloader, net2, criterion, len(trset), args)
    tr_acc, tr_conf = print_acc_conf(selena_infer_train_conf_tr, train_label_tr_attack)
    print("te set")
    te_loss, selena_infer_train_conf_te = selena_test(teloader, net2, criterion, len(teset), args)
    te_acc, te_conf = print_acc_conf(selena_infer_train_conf_te, train_label_te_attack)
    print("test set")
    test_loss, selena_infer_test_conf = selena_test(testloader, net2, criterion, len(testset), args)
    test_acc, test_conf = print_acc_conf(selena_infer_test_conf, test_label)
    print("reference set")
    ref_loss, selena_infer_ref_conf = selena_test(refloader, net2, criterion, len(refset), args)
    ref_acc, ref_conf = print_acc_conf(selena_infer_ref_conf, ref_label)

    print("For comparison on final selena output")
    print("avg acc  on train/all test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_acc, all_test_acc ,tr_acc, te_acc, test_acc, ref_acc))
    print("avg conf on train/all_test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_conf, all_test_conf, tr_conf, te_conf, test_conf, ref_conf))

    system_attack(selena_infer_train_conf_tr, train_label_tr_attack, selena_infer_train_conf_te, train_label_te_attack, selena_infer_ref_conf, ref_label, selena_infer_test_conf, test_label, num_class=num_class,attack_epochs=attack_epochs,batch_size=batch_size)

if __name__ == '__main__':
    main()
