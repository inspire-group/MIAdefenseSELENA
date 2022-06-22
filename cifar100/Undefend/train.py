import argparse
import os
import shutil
import random
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
from cifar_utils import transform_train, transform_test, Cifardata, DistillCifardata, WarmUpLR, ModelwNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(trainloader, model, criterion, optimizer, warmup_scheduler, epoch, args):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()


    for batch_ind, (inputs, targets) in enumerate(trainloader):
        if epoch <= args.warmup:
            warmup_scheduler.step()

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)

def test(testloader, model, criterion):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, acc, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

def main():
    parser = argparse.ArgumentParser(description='setting for cifar100')
    parser.add_argument('--classifier_epochs', type=int, default=200, help='classifier epochs')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')    
    parser.add_argument('--print_epoch', type=int, default=5, help='print model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--warmup', type=int, default=1, help='warm up epochs')
    parser.add_argument('--num_worker', type=int, default=1, help='number workers')
    parser.add_argument('--num_class', type=int, default=100, help='num class')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    classifier_epochs = args.classifier_epochs
    print_epoch = args.print_epoch
    warmup = args.warmup
    num_worker = args.num_worker

    DATASET_PATH = os.path.join(root_dir, 'cifar100',  'data')
    checkpoint_path = os.path.join(root_dir, 'cifar100', 'checkpoints', 'undefend')
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


    trainset = Cifardata(train_data, train_label, transform_train)
    traintestset = Cifardata(train_data, train_label, transform_test)
    testset = Cifardata(test_data, test_label, transform_test)
    refset = Cifardata(ref_data, ref_label, transform_test)

    trset = Cifardata(train_data_tr_attack, train_label_tr_attack, transform_test)
    teset = Cifardata(train_data_te_attack, train_label_te_attack, transform_test)
    alltestset = Cifardata(all_test_data, all_test_label, transform_test)

    trloader = torch.utils.data.DataLoader(trset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    alltestloader = torch.utils.data.DataLoader(alltestset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    traintestloader = torch.utils.data.DataLoader(traintestset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)


    best_acc = 0.00
    model_1 = resnet18()
    model = ModelwNorm(model_1)

    criterion = nn.CrossEntropyLoss()
    model = model.to(device, torch.float)
    criterion = criterion.to(device, torch.float)

    iter_per_epoch = len(trainloader)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [60, 120, 160], gamma=0.2) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer,  iter_per_epoch * warmup)

    print("training sets: {:d}".format(len(trainset)))

    best_epoch = 0
    for epoch in range(1, classifier_epochs+1):
        if epoch > 1:
            train_scheduler.step(epoch)
  
        training_loss, training_acc = train(trainloader, model, criterion, optimizer, warmup_scheduler, epoch, args)

        train_loss, train_acc  = test(traintestloader, model, criterion)
        test_loss, test_acc = test(testloader, model, criterion)


        # save model
        is_best = test_acc>best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            best_epoch = epoch

        save_checkpoint({
                    'epoch': epoch ,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, best_acc, checkpoint=checkpoint_path, filename='Depoch%d.pth.tar'%(epoch))
        #if (epoch)%print_epoch ==0:
        lr = get_learning_rate(optimizer)
        print('Epoch: [{:d} | {:d}]: learning rate:{:.4f}. acc: training|train|test: {:.4f}|{:.4f}|{:.4f}. loss: training|train|test: {:.4f}|{:.4f}'.format(epoch, classifier_epochs, lr[0], training_acc, train_acc, test_acc, training_loss, train_loss, test_loss))
        sys.stdout.flush()
    print("best acc: {:.4f}".format(best_acc))

    print("Final saved epoch {:d} acc: {:.4f}".format(best_epoch, best_acc))

if __name__ == '__main__':
        main()