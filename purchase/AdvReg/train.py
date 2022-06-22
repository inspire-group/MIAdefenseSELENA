import argparse
import os
import shutil
import sys
import yaml
import random
import numpy as np

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
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from purchase import PurchaseClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(100,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )          
        self.labels=nn.Sequential(
#            nn.Linear(num_classes, 512),
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
                nn.init.normal_(self.state_dict()[key], std=0.01)               
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()

    def forward(self, x1, l):
        out_x1 = self.features(x1)        
        out_l = self.labels(l)            
        is_member =self.combine(torch.cat((out_x1,out_l),1))        
        return self.output(is_member)

def train(train_data, train_labels, model, criterion, optimizer, batch_size):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t =  int(np.ceil(len(train_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min((batch_ind+1)*batch_size, len(train_data))
        inputs = train_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = train_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item()/100.0, inputs.size()[0])
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, top1.avg)

def train_privatly(train_data, train_labels,  model, inference_model, criterion, optimizer, batch_size, num_batchs=10000, skip_batch = 0, alpha = 1):
    model.train()
    inference_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    len_t = int(np.ceil(len(train_data)/batch_size))

    for batch_ind in range(len_t):
        if batch_ind < skip_batch:
            continue
        if batch_ind > skip_batch + num_batchs:
            break

        end_idx = min((batch_ind+1)*batch_size, len(train_data))
        inputs = train_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = train_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)

        one_hot_tr = torch.zeros(outputs.size()[0], outputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, targets.to(device, torch.long).view([-1,1]).data, 1)
         
        inference_output = inference_model(outputs, target_one_hot).view([-1])
        loss = criterion(outputs, targets) + ((alpha)*(torch.mean((inference_output)) - 0.5))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item()/100.0, inputs.size()[0])
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, top1.avg)


def privatly_attack(train_data, train_labels, attack_data, attack_labels, model, inference_model, inference_criterion, inference_optimizer, batch_size, num_batchs=100000, skip_batch=0, is_train = True):
    model.eval()

    if is_train:
        inference_model.train()
    else:
        inference_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    len_data = min(len(train_data), len(attack_data))
    len_t =  int(np.ceil(len_data/batch_size))

    for batch_ind in range(len_t):
        if batch_ind >= skip_batch+num_batchs:
            break
        if batch_ind < skip_batch:
            continue

        end_idx = min((batch_ind+1)*batch_size,  len_data)
        inputs = train_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = train_labels[batch_ind*batch_size: end_idx].to(device, torch.long)
        inputs_attack = attack_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets_attack = attack_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        outputs = (model(inputs)).detach().cpu().numpy()
        outputs_non = (model(inputs_attack)).detach().cpu().numpy()

        comb_input = np.zeros((outputs.shape[0]+outputs_non.shape[0], outputs.shape[1]))#torch.cat((outputs,outputs_non))
        comb_input[:outputs.shape[0], :] = outputs
        comb_input[outputs.shape[0]:, :] = outputs_non
        comb_inputs = (torch.from_numpy(comb_input).type(torch.FloatTensor)).to(device, torch.float)

        comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).to(device, torch.float)
  
        one_hot_tr = torch.zeros(comb_inputs.size()[0],comb_inputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).data, 1)

        attack_output = inference_model(comb_inputs, target_one_hot).view([-1])

        att_labels = torch.zeros((inputs.size()[0]+inputs_attack.size()[0]))
        att_labels [:inputs.size()[0]] =1.0
        att_labels [inputs.size()[0]:] =0.0
        is_member_labels = att_labels.to(device, torch.float)
        
        loss_attack = inference_criterion(attack_output, is_member_labels)
        
        if is_train:
            inference_optimizer.zero_grad()
            loss_attack.backward()
            inference_optimizer.step()

        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy()>0.5), (is_member_labels.data.cpu().numpy()> 0.5)))        

        losses.update(loss_attack.item(), comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])

    return (losses.avg, top1.avg)

def test(test_data, labels, model, criterion, batch_size):

    losses = AverageMeter()
    top1 = AverageMeter()
    confs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    len_t = int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):

        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        
        infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
        conf = np.mean(np.max(infer_np[batch_ind*batch_size:end_idx], axis = 1))
        confs.update(conf, inputs.size()[0])        

        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item()/100.0, inputs.size()[0])

        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, top1.avg, confs.avg)


def advregtest(test_data, labels, model, criterion, batch_size):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()

    len_t = int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
        
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, infer_np)

def save_checkpoint(state, is_best, checkpoint, filename):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def main():
    parser = argparse.ArgumentParser(description = 'Setting for Purchase datataset')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--classifier_epochs', type = int, default = 30, help = 'classifier epochs')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')
    parser.add_argument('--alpha', type=float, default=3, help= 'regularzation param')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    alpha = args.alpha
    attack_epochs = args.attack_epochs
    classifier_epochs = args.classifier_epochs

    DATASET_PATH = os.path.join(root_dir, 'purchase',  'data')
    checkpoint_path = os.path.join(root_dir, 'purchase', 'checkpoints', 'AdvReg')
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

    model = PurchaseClassifier().to(device,torch.float)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    saved_epoch = 0
    best_acc = 0.0

    train_data_tr_attack_tensor = torch.from_numpy(train_data_tr_attack).type(torch.FloatTensor)
    train_label_tr_attack_tensor = torch.from_numpy(train_label_tr_attack).type(torch.LongTensor)
    train_data_te_attack_tensor = torch.from_numpy(train_data_te_attack).type(torch.FloatTensor)
    train_label_te_attack_tensor = torch.from_numpy(train_label_te_attack).type(torch.LongTensor)

    ref_data_used = ref_data.copy()
    ref_label_used = ref_label.copy()

    attack_model0 = InferenceAttack_HZ(num_class).to(device, torch.float)
    attack_criterion0 = nn.MSELoss().to(device, torch.float)
    attack_optimizer0 = optim.Adam(attack_model0.parameters(),lr=0.0001)

    for epoch in range(1, classifier_epochs+1):

        r= np.arange(len(train_data))
        np.random.shuffle(r)
        train_data = train_data[r]
        train_label = train_label[r]
        train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

        r = np.arange(len(ref_data_used))
        np.random.shuffle(r)
        ref_data_used = ref_data_used[r]
        ref_label_used = ref_label_used[r]
        ref_data_tensor = torch.from_numpy(ref_data_used).type(torch.FloatTensor)
        ref_label_tensor = torch.from_numpy(ref_label_used).type(torch.LongTensor)

        if epoch == 1:
            training_loss, trainng_acc = train(train_data_tensor, train_label_tensor, model, criterion, optimizer, batch_size)

            for i in range(5):
                _, at_acc = privatly_attack(train_data_tensor, train_label_tensor, ref_data_tensor, ref_label_tensor, model, attack_model0, attack_criterion0, attack_optimizer0, batch_size)
                #print(at_acc)

        else:
            for i in range((len(train_data)//(batch_size*1))):
                _, at_acc = privatly_attack(train_data_tensor, train_label_tensor, ref_data_tensor, ref_label_tensor, model, attack_model0, attack_criterion0, attack_optimizer0, batch_size, 13*(int(512/batch_size)), (i*13*(int(512/batch_size)))%(int(np.ceil(len(train_data)/batch_size))))
                _, _ = train_privatly(train_data_tensor, train_label_tensor, model, attack_model0, criterion, optimizer, batch_size, 1, (i*1)%((len(train_data)//batch_size)), alpha)


        train_loss, train_acc, train_conf = test(train_data_tensor, train_label_tensor, model, criterion, batch_size)
        test_loss, test_acc, test_conf = test(test_data_tensor,test_label_tensor, model, criterion, batch_size)

        # save model
        is_best = test_acc>best_acc
        best_acc = max(test_acc, best_acc)

        if is_best:
            saved_epoch = epoch

        save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=checkpoint_path, filename='Depoch%d.pth.tar'%(epoch))

        print('Epoch: [{:d} | {:d}]: acc: train|test: {:.4f}|{:.4f}. conf: train|test: {:.4f}|{:.4f}'.format(epoch, classifier_epochs, train_acc, test_acc, train_conf, test_conf))
        sys.stdout.flush()

    print("Final saved epoch: {:d}. ".format(saved_epoch))

if __name__ == '__main__':
    main()