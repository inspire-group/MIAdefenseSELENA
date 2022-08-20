import argparse
import os
import sys
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
from torch.optim.lr_scheduler import _LRScheduler

config_file = './../../../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'attack'))
sys.path.append(os.path.join(src_dir, 'models'))
from adaptive_attack import system_attack

from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from resnet import resnet18
from cifar_utils import transform_train, transform_test, Cifardata, DistillCifardata, WarmUpLR, ModelwNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splitai_test(testloader, model, criterion, len_data, ckpt_path, non_model_indices_all, mode, args):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()

    num_class = args.num_class
    batch_size = args.batch_size
    split_model = args.K
    non_model = args.L

    infer_np = np.zeros((len_data, num_class))
    cnt = np.zeros(split_model)
    corr = np.zeros(split_model)
    conf = np.zeros(split_model)

    for batch_ind, (inputs, targets) in enumerate(testloader):
        # compute output
        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs_np = np.zeros((inputs.shape[0], num_class))
        outputs_conf = np.zeros((split_model, inputs.shape[0]))
        tmp_outputs_np = np.zeros((split_model, inputs.shape[0], num_class))

        for model_ind in range(split_model):
            ckpt = torch.load(os.path.join(ckpt_path, str(model_ind), 'model_best.pth.tar'))
            model.load_state_dict(ckpt['state_dict'])
            model = model.to(device,torch.float)
            model.eval()
            with torch.no_grad():
                tmp_outputs = (F.softmax(model(inputs),dim=1)).detach().cpu().numpy()
            tmp_outputs_np[model_ind,:,:] = tmp_outputs

            outputs_conf[model_ind,:] = np.max(tmp_outputs, axis = 1)
        
        temp = np.zeros(split_model)
        for ind in range(inputs.shape[0]):

            if mode == 1:
                if non_model == 1:
                    outputs_np[ind,:] = tmp_outputs_np[(targets[ind, 1].detach().cpu().numpy().astype(np.int32)), ind, :]
                else:                
                    outputs_np[ind,:] = np.mean(tmp_outputs_np[(targets[ind, 1:].detach().cpu().numpy().astype(np.int32)), ind, :], axis = 0)
            
            elif mode ==2:
                rand_ind = np.random.randint(non_model_indices_all.shape[0])

                if non_model == 1:
                    outputs_np[ind,:] = tmp_outputs_np[non_model_indices_all[rand_ind, 1:].astype(np.int32), ind, :]
                else:                
                    outputs_np[ind,:] = np.mean(tmp_outputs_np[non_model_indices_all[rand_ind, 1:].astype(np.int32), ind, :], axis = 0)


            tmp_rank = np.argsort(outputs_conf[:,ind])
            if mode == 1:
                for kidx in range(split_model):
                    if tmp_rank[kidx] in targets[ind, -non_model:]:
                        cnt[kidx] = cnt[kidx] + 1

                for model_ind in range(split_model):
                    if np.argmax(tmp_outputs_np[tmp_rank[model_ind],ind,:]) == int(targets[ind, 0]):
                        corr[model_ind] = corr[model_ind] + 1
                    temp[model_ind] = temp[model_ind] + outputs_conf[tmp_rank[model_ind], ind]

            if mode == 2:
                for model_ind in range(split_model):
                    if np.argmax(tmp_outputs_np[tmp_rank[model_ind],ind,:]) == int(targets[ind]):
                        corr[model_ind] = corr[model_ind] + 1
                    temp[model_ind] = temp[model_ind] + outputs_conf[tmp_rank[model_ind], ind]

        for model_ind in range(split_model):
            conf[model_ind] = conf[model_ind] + temp[model_ind]
        
        infer_np[batch_ind*batch_size: batch_ind*batch_size+inputs.shape[0]] = outputs_np
        outputs = torch.from_numpy(outputs_np).to(device, torch.float)        
        if mode == 1:
            loss = criterion(outputs, targets[:,0])
        if mode == 2:
            loss = criterion(outputs, targets)

    print(len_data)
    if mode == 1:
        for ind in range(split_model):
            print("{:d} least confidence : matches as non-training data {:d}/{:.4f}. corr: {:d}/{:.4f}. conf_avg: {:.4f}".format(ind+1, int(cnt[ind]), cnt[ind]*1.0/len_data, int(corr[ind]), corr[ind]*1.0/len_data, conf[ind]/len_data))
    elif mode == 2:
        for ind in range(split_model):
            print("{:d} least confidence: corr: {:d}/{:.4f}. conf_avg: {:.4f}".format(ind+1, int(corr[ind]), corr[ind]*1.0/len_data, conf[ind]/len_data))
    return (losses.avg, infer_np)

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


def repeat_arrays(infer_conf, train_label, split_ai_conf, total_num):
    if len(train_label)==total_num:
        return_label = train_label.copy()
        return_infer = infer_conf.copy()
        return_split_ai = split_ai_conf.copy()

    else:
        total_int = total_num//len(train_label)
        return_infer = infer_conf.copy()
        return_label = train_label.copy()
        return_split_ai = split_ai_conf.copy()
        for i in range(total_int-1):
            return_infer =  np.concatenate((return_infer, infer_conf), axis=0)
            return_label = np.concatenate((return_label, train_label), axis=0)
            return_split_ai = np.concatenate((return_split_ai, split_ai_conf), axis=0)

    return return_infer, return_label, return_split_ai

def main():
    parser = argparse.ArgumentParser(description='setting for cifar100')
    parser.add_argument('--K', type=int, default=25, help='total sub-models in split-ai')
    parser.add_argument('--L', type=int, default=10, help='non_model for each sample in split-ai')
    parser.add_argument('--attack_epochs', type=int, default=150, help='attack epochs in NN attack')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_worker', type=int, default=1, help='number workers')
    parser.add_argument('--num_class', type=int, default=100, help='num class')
    parser.add_argument('--known_ratio', type=float, default=0.5, help='known ratio of member samples by attacker')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    split_model = args.K
    non_model = args.L
    attack_epochs = args.attack_epochs
    batch_size = args.batch_size
    num_class = args.num_class
    load_name = str(split_model) + '_' + str(non_model)
    num_worker = args.num_worker

    DATASET_PATH = os.path.join(root_dir, 'cifar100',  'data')
    checkpoint_path = os.path.join(root_dir, 'cifar100', 'checkpoints', 'K_L', load_name)
    checkpoint_path_shadow= os.path.join(checkpoint_path, 'shadow', str(args.known_ratio))
    checkpoint_path_selena = os.path.join(checkpoint_path, 'selena')
    print(checkpoint_path_shadow, checkpoint_path_selena)

    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'attacker', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'K_L', load_name, 'attacker', 'te_label.npy'))
    full_train_data = np.concatenate((train_data_tr_attack, train_data_te_attack), axis = 0)
    full_train_label = np.concatenate((train_label_tr_attack, train_label_te_attack), axis = 0)
    test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'))
    test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))
    ref_data = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'))
    ref_label = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'))
    all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))

    len_train_data = len(full_train_label)
    known_num = int(args.known_ratio*len_train_data)
    unknown_num = len_train_data - known_num
    attack_know_data = full_train_data[:known_num, :]
    attack_know_label = full_train_label[:known_num]
    attack_unknow_data = full_train_data[known_num:, :]
    attack_unknow_label = full_train_label[known_num:]

    #print first 20 labels for each subset, for checking with other experiments
    print(train_label_tr_attack[:20, 0])
    print(train_label_te_attack[:20, 0])
    print(test_label[:20])
    print(ref_label[:20])

    attack_unknow_label = attack_unknow_label[:, 0]
    raw_train_data = attack_know_label.copy()

    knowset = Cifardata(attack_know_data, attack_know_label, transform_test)
    unknowset = Cifardata(attack_unknow_data, attack_unknow_label, transform_test)
    testset = Cifardata(test_data, test_label, transform_test)
    refset = Cifardata(ref_data, ref_label, transform_test)

    knowloader = torch.utils.data.DataLoader(knowset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    unknowloader = torch.utils.data.DataLoader(unknowset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    print("Attack Training: # of train data: {:d}, # of ref data: {:d}".format(int(len(attack_know_label)), len(ref_data)))
    print("Attack Testing: # of train data: {:d}, # of test data: {:d}".format(int(len(attack_unknow_label)), len(test_data)))
    criterion = (nn.CrossEntropyLoss()).to(device, torch.float)

    net_t = resnet18()
    net = ModelwNorm(net_t)
    net = net.to(device, torch.float)

    train_mode = 1
    test_mode = 2

    print("attacker know set")
    know_loss, infer_attack_know = splitai_test(knowloader, net, criterion, len(knowset), checkpoint_path_shadow, raw_train_data, train_mode, args)
    know_acc, know_conf = print_acc_conf(infer_attack_know, attack_know_label[:,0])
    print("attack unknow set")
    unknow_loss, infer_attack_unknow = splitai_test(unknowloader, net, criterion, len(unknowset), checkpoint_path_shadow, raw_train_data, test_mode, args)
    unknow_acc, unknow_conf = print_acc_conf(infer_attack_unknow, attack_unknow_label)
    print("test set")
    test_loss, infer_test_conf = splitai_test(testloader, net, criterion, len(testset), checkpoint_path_shadow, raw_train_data, test_mode, args)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference set")
    ref_loss, infer_ref_conf = splitai_test(refloader, net, criterion, len(refset), checkpoint_path_shadow, raw_train_data, test_mode, args)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)
    
    print("For comparison on attacker shadow split-ai output")
    print("avg acc  on know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_acc,  unknow_acc, test_acc, ref_acc))
    print("avg conf on know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_conf, unknow_conf, test_conf, ref_conf))
    
    net2_t = resnet18()
    net2 = ModelwNorm(net2_t)
    net2 = net2.to(device, torch.float)

    resume = checkpoint_path_selena +'/model_best.pth.tar'
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net2.load_state_dict(checkpoint['state_dict'])
    net2 = net2.to(device, torch.float)

    attack_know_label = attack_know_label[:, 0]
    knowset = Cifardata(attack_know_data, attack_know_label, transform_test)
    knowloader = torch.utils.data.DataLoader(knowset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    
    print("selena output labels")
    print("attacker known set")
    know_loss, selena_infer_know_conf = selena_test(knowloader, net2, criterion, len(knowset), args)
    know_acc, know_conf = print_acc_conf(selena_infer_know_conf, attack_know_label)
    print("attacker unknown set")
    unknow_loss, selena_infer_unknow_conf = selena_test(unknowloader, net2, criterion, len(unknowset), args)
    unknow_acc, unknow_conf = print_acc_conf(selena_infer_unknow_conf, attack_unknow_label)
    print("test set")
    test_loss, selena_infer_test_conf = selena_test(testloader, net2, criterion, len(testset), args)
    test_acc, test_conf = print_acc_conf(selena_infer_test_conf, test_label)
    print("reference set")
    ref_loss, selena_infer_ref_conf = selena_test(refloader, net2, criterion, len(refset), args)
    ref_acc, ref_conf = print_acc_conf(selena_infer_ref_conf, ref_label)
    
    print("For comparison on final selena output")
    print("avg acc  on know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_acc,  unknow_acc, test_acc, ref_acc))
    print("avg conf on know/unknow/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(know_conf, unknow_conf, test_conf, ref_conf))

    selena_infer_ref_conf_eval, ref_label_eval, infer_ref_conf_eval = repeat_arrays(selena_infer_ref_conf, ref_label, infer_ref_conf, known_num)
    selena_infer_test_conf_eval, test_label_eval, infer_test_conf_eval = repeat_arrays(selena_infer_test_conf, test_label, infer_test_conf, unknown_num)

    system_attack(infer_attack_know, selena_infer_know_conf, attack_know_label, infer_attack_unknow, selena_infer_unknow_conf, attack_unknow_label, infer_ref_conf_eval, selena_infer_ref_conf_eval, ref_label_eval, infer_test_conf_eval, selena_infer_test_conf_eval, test_label_eval, num_class,attack_epochs=attack_epochs,batch_size=batch_size)

if __name__ == '__main__':
    main()
