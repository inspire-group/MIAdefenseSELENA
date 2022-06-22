import argparse
import os
import numpy as np
import sys
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

config_file = './../../env.yml'
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
from cifar_utils import transform_train, transform_test, DistillCifardata, WarmUpLR, ModelwNorm, Cifardata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def advregtest(testloader, model, criterion, len_data, args):
    # switch to evaluate mode
    model.eval()
    batch_size = args.batch_size
    num_class = args.num_class

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

    return (losses.avg, infer_np)

def main():
    parser = argparse.ArgumentParser(description='Setting for CIFAR100 by Adversarial Regularization')
    parser.add_argument('--classifier_epochs',type=int,default=200,help='classifier epochs')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')    
    parser.add_argument('--print_epoch', type=int, default=5, help='print model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--warmup', type=int, default=1, help='warm up epochs')
    parser.add_argument('--num_worker', type=int, default=1, help='number workers')
    parser.add_argument('--num_class', type=int, default=100, help='num class')
    parser.add_argument('--alpha', type=float, default=6, help='para for Adversarial Regularization')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    classifier_epochs = args.classifier_epochs
    print_epoch = args.print_epoch
    warmup = args.warmup
    num_worker = args.num_worker
    alpha = args.alpha

    DATASET_PATH = os.path.join(root_dir, 'cifar100',  'data')
    checkpoint_path = os.path.join(root_dir, 'cifar100', 'checkpoints', 'AdvReg')
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

    r = np.arange(len(train_data))
    np.random.shuffle(r)
    trainset = Cifardata(train_data, train_label, transform_train)
    traintestset = Cifardata(train_data, train_label, transform_test)

    testset = Cifardata(test_data, test_label, transform_test)
    refset = Cifardata(ref_data, ref_label, transform_test)
    alltestset = Cifardata(all_test_data, all_test_label, transform_test)
    trset = Cifardata(train_data_tr_attack, train_label_tr_attack, transform_test)
    teset = Cifardata(train_data_te_attack, train_label_te_attack, transform_test)

    traintestloader = torch.utils.data.DataLoader(traintestset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    alltestloader = torch.utils.data.DataLoader(alltestset, batch_size = batch_size, shuffle = False, num_workers = num_worker)
    trloader = torch.utils.data.DataLoader(trset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    criterion = (nn.CrossEntropyLoss()).to(device, torch.float)

    net_1 = resnet18()
    net = ModelwNorm(net_1)
    resume = checkpoint_path +'/model_best.pth.tar'
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device, torch.float)

    print("Attack Training: # of train data: {:d}, # of ref data: {:d}".format(int(len(train_data_tr_attack)), len(ref_data)))
    print("Attack Testing: # of train data: {:d}, # of test data: {:d}".format(int(len(train_data_te_attack)), len(test_data)))

    print("training sets")
    train_loss, infer_train_conf = advregtest(traintestloader, net, criterion, len(traintestset), args)
    train_acc, train_conf = print_acc_conf(infer_train_conf, train_label)
    print("all test sets")
    all_test_loss, infer_all_test_conf = advregtest(alltestloader, net, criterion, len(alltestset), args)
    all_test_acc, all_test_conf = print_acc_conf(infer_all_test_conf, all_test_label)

    print("tr sets")
    tr_loss, infer_train_conf_tr = advregtest(trloader, net, criterion, len(trset), args)
    tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack)
    print("te sets")
    te_loss, infer_train_conf_te = advregtest(teloader, net, criterion, len(teset), args)
    te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack)

    print("test sets")
    test_loss, infer_test_conf = advregtest(testloader, net, criterion, len(testset), args)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference sets")
    ref_loss, infer_ref_conf, = advregtest(refloader, net, criterion, len(refset), args)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    print("Adversarial Regularization outputs:")
    print("avg acc on  training/all test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(train_acc, all_test_acc, tr_acc, te_acc, test_acc, ref_acc))
    print("avg conf on training/all test/tr/te/test/reference set {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(train_conf, all_test_conf, tr_conf, te_conf, test_conf, ref_conf))

    system_attack(infer_train_conf_tr, train_label_tr_attack, infer_train_conf_te, train_label_te_attack, infer_ref_conf, ref_label, infer_test_conf, test_label, num_class,attack_epochs=attack_epochs,batch_size=256)

if __name__ == '__main__':
    main()
