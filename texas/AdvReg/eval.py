import argparse
import os
import sys
import yaml
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
from binary_flip_noise_attack import flip_noise_attack
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from texas import TexasClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
    parser = argparse.ArgumentParser(description = 'Setting for Texas datataset')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')
    parser.add_argument('--alpha', type=float, default=2, help= 'regularzation param')
    parser.add_argument('--flip_range', type = int, default = 100, help = 'flip range')
    parser.add_argument('--nruns', type = int, default = 100, help = 'repeated runs for a fixed flip range ')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    alpha = args.alpha
    attack_epochs = args.attack_epochs
    flip_range = args.flip_range
    nruns = args.nruns

    DATASET_PATH = os.path.join(root_dir, 'texas',  'data')
    checkpoint_path = os.path.join(root_dir, 'texas', 'checkpoints', 'AdvReg')
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

    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    train_data_tr_attack_tensor = torch.from_numpy(train_data_tr_attack).type(torch.FloatTensor)
    train_label_tr_attack_tensor = torch.from_numpy(train_label_tr_attack).type(torch.LongTensor)
    train_data_te_attack_tensor = torch.from_numpy(train_data_te_attack).type(torch.FloatTensor)
    train_label_te_attack_tensor = torch.from_numpy(train_label_te_attack).type(torch.LongTensor)

    ref_data_used = ref_data.copy()
    ref_label_used = ref_label.copy()


    net2 = TexasClassifier()
    resume = checkpoint_path + '/Depoch4.pth.tar'
    print('\n\n==> Resuming from checkpoint: '+resume)
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net2.load_state_dict(checkpoint['state_dict'])
    net2 = net2.to(device, torch.float)

    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)

    print("training tr set")
    tr_loss, infer_train_conf_tr = advregtest(train_data_tr_attack_tensor, train_label_tr_attack_tensor, net2, criterion, batch_size)
    tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack)
    print("training te set")
    te_loss, infer_train_conf_te = advregtest(train_data_te_attack_tensor, train_label_te_attack_tensor, net2, criterion, batch_size)
    te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack)

    print("test set")
    test_loss, infer_test_conf = advregtest(test_data_tensor, test_label_tensor, net2, criterion, batch_size)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference set")
    ref_loss, infer_ref_conf = advregtest(ref_data_tensor, ref_label_tensor, net2, criterion, batch_size)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    print("For comparison on AdvReg output")
    print("avg acc on: tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(tr_acc, te_acc, test_acc, ref_acc))
    print("avg conf on tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(tr_conf, te_conf, test_conf, ref_conf))
    system_attack(infer_train_conf_tr, train_label_tr_attack, infer_train_conf_te, train_label_te_attack, infer_ref_conf, ref_label, infer_test_conf, test_label, num_class,attack_epochs=attack_epochs,batch_size=batch_size)
    flip_noise_attack(net2, train_data_tr_attack, train_label_tr_attack, train_data_te_attack, train_label_te_attack, ref_data, ref_label, test_data, test_label, num_class, batch_size,flip_range=flip_range,repeated=nruns)

if __name__ == '__main__':
    main()