import argparse
import os
import random
import numpy as np
import sys
import yaml
from PIL import Image

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
from cifar_utils import transform_train, transform_test, DistillCifardata, WarmUpLR, ModelwNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CifarAttack(data.Dataset):
    def __init__(self, data, labels, non_data, non_labels, transform):
        self.labels = labels
        self.data = data
        self.non_labels = non_labels
        self.non_data = non_data
        self.transform = transform
        self.min_len = min(len(labels), len(non_labels))
        if len(labels)>len(non_labels):
            self.flag = 0
        else:
            self.flag = 1

    def __getitem__(self, index):
        if index>0 and index%self.min_len==0:
            if self.flag ==0:
                r = np.arange(len(self.non_labels))
                np.random.shuffle(r)
                self.non_labels = self.non_labels[r]
                self.non_data = self.non_data[r]
            else:
                r = np.arange(len(self.labels))
                np.random.shuffle(r)
                self.labels = self.labels[r]
                self.data = self.data[r]

        if self.flag ==0:
            index2 = index%self.min_len
            index1 = index
        else:
            index1 = index%self.min_len
            index2= index
        label = self.labels[index1]
        img =  Image.fromarray((self.data[index1].transpose(1,2,0).astype(np.uint8)))
        img = self.transform(img)

        non_label = self.non_labels[index2]
        non_img =  Image.fromarray((self.non_data[index2].transpose(1,2,0).astype(np.uint8)))
        non_img = self.transform(non_img)

        return img, label, non_img, non_label

    def __len__(self):
        return max(len(self.labels), len(self.non_labels))

    def update(self):
        r = np.arange(len(self.labels))
        np.random.shuffle(r)
        self.labels = self.labels[r]
        self.data = self.data[r]

        r = np.arange(len(self.non_labels))
        np.random.shuffle(r)
        self.non_labels = self.non_labels[r]
        self.non_data = self.non_data[r]


class Cifardata(data.Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        img =  Image.fromarray((self.data[index].transpose(1,2,0).astype(np.uint8)))
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

    def update(self):
        r = np.arange(len(self.labels))
        np.random.shuffle(r)
        self.labels = self.labels[r]
        self.data = self.data[r]

class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*2,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()

    def forward(self, x1, l):
        out_x1 = self.features(x1)
        out_l = self.labels(l)
        is_member =self.combine( torch.cat((out_x1,out_l),1))
        
        return self.output(is_member)

def train(trainloader, model, criterion, optimizer, epoch, warmup_scheduler, args):
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

        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)

def train_privatly(trainloader, model, inference_model, criterion, optimizer, epoch, warmup_scheduler, args, num_batchs=10000, skip_batch = 0, alpha = 1):
    model.train()
    inference_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):        
        if batch_ind < skip_batch:
            continue
        if batch_ind > skip_batch + num_batchs:
            break

        if epoch <= args.warmup:
            warmup_scheduler.step()

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)

        one_hot_tr = torch.zeros(outputs.size()[0], outputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, targets.to(device, torch.long).view([-1,1]).data, 1)
         
        inference_output = inference_model(outputs, target_one_hot).view([-1])
        loss = criterion(outputs, targets) + ((alpha)*(torch.mean((inference_output)) - 0.5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)

def train_privatly_attack(attacktrainloader, model, inference_model, inference_criterion, inference_optimizer, num_batchs=100000, skip_batch=0):
    # switch to train mode
    inference_model.train()
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_ind, (inputs, targets, inputs_attack, targets_attack) in enumerate(attacktrainloader):
        
        if batch_ind >= skip_batch+num_batchs:
            break
        if batch_ind < skip_batch:
            continue

        targets = targets.to(device, torch.long)
        targets_attack = targets_attack.to(device, torch.long)
        inputs = inputs.to(device, torch.float)
        inputs_attack = inputs_attack.to(device, torch.float)

        outputs = model(inputs)
        outputs_non = model(inputs_attack)        

        comb_inputs = torch.cat((outputs,outputs_non))
        comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).to(device, torch.float)
  
        one_hot_tr = torch.zeros(comb_inputs.size()[0],comb_inputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).data, 1)

        attack_output = inference_model(comb_inputs, target_one_hot).view([-1])

        att_labels = torch.zeros((inputs.size()[0]+inputs_attack.size()[0]))
        att_labels [:inputs.size()[0]] =1.0
        att_labels [inputs.size()[0]:] =0.0
        is_member_labels = att_labels.to(device, torch.float)
        
        loss_attack = inference_criterion(attack_output, is_member_labels)
        inference_optimizer.zero_grad()
        loss_attack.backward()
        inference_optimizer.step()

        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy()>0.5), (is_member_labels.data.cpu().numpy()> 0.5)))        

        losses.update(loss_attack.item(), comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, batch_size):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))   
        top1.update(prec1.item()/100.0, inputs.size()[0])

        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, acc, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        filepath = os.path.join(checkpoint, 'model_best.pth.tar')
        if os.path.exists(filepath):
            tmp_ckpt = torch.load(filepath)
            best_acc = tmp_ckpt['best_acc']
            if best_acc > acc:
                return
        torch.save(state, filepath)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

def main():
    parser = argparse.ArgumentParser(description='Setting for CIFAR100 by Adversarial Regularization')
    parser.add_argument('--classifier_epochs',type=int,default=200,help='classifier epochs')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')    
    parser.add_argument('--print_epoch', type=int, default=5, help='print model training stats per print_epoch_splitai during splitai training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--warmup', type=int, default=1, help='warm up epochs')
    parser.add_argument('--num_worker', type=int, default=2, help='number workers')
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    traintestloader = torch.utils.data.DataLoader(traintestset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    alltestloader = torch.utils.data.DataLoader(alltestset, batch_size = batch_size, shuffle = False, num_workers = num_worker)
    trloader = torch.utils.data.DataLoader(trset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False, num_workers=num_worker)


    best_acc = 0.00
    model_1 = resnet18()
    model = ModelwNorm(model_1)
    criterion = (nn.CrossEntropyLoss()).to(device, torch.float)
    model = model.to(device, torch.float)
    print("training sets: {:d}".format(len(train_data)))

    iter_per_epoch = len(trainloader)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [60, 120, 160], gamma=0.2) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer,  iter_per_epoch * args.warmup)

    attackset = CifarAttack(train_data, train_label, all_test_data, all_test_label, transform_test)
    attackloader = torch.utils.data.DataLoader(attackset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    print("attack set: {:d}".format(len(attackset)))

    best_epoch = 0

    attack_model0 = InferenceAttack_HZ(num_class).to(device, torch.float)
    attack_criterion0 = nn.MSELoss().to(device, torch.float)
    attack_optimizer0 = optim.Adam(attack_model0.parameters(),lr=0.0001)

    for epoch in range(1, classifier_epochs+1):

        trainset.update()
        attackset.update()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
        attackloader = torch.utils.data.DataLoader(attackset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

        if epoch > 1:
            train_scheduler.step(epoch)


        if epoch <= 4:
            training_loss, training_acc = train(trainloader, model, criterion, optimizer, epoch, warmup_scheduler, args)

        else:
            for i in range(25000//batch_size):
                _, _, = train_privatly_attack(attackloader, model, attack_model0, attack_criterion0, attack_optimizer0, 1,  (i*1)%(len(attackset)//batch_size))

                _, _, = train_privatly(trainloader, model, attack_model0, criterion, optimizer, epoch, warmup_scheduler, args, 1, (i*1)%(len(trainset)//batch_size), alpha)


        test_loss, test_acc = test(testloader, model, criterion, batch_size)
 
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
        print('Epoch: [{:d} | {:d}]: learning rate:{:.4f}. acc: test: {:.4f}. loss: test: {:.4f}'.format(epoch, classifier_epochs, lr[0], test_acc, test_loss))
        sys.stdout.flush()

    print("Final saved epoch {:d} with best acc {:.4f}".format(best_epoch, best_acc))

if __name__ == '__main__':
    main()
