import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

small_delta=1e-30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
        
def train_attack(infer_data, labels, attack_infer_data, attack_labels, attack_model, attack_criterion, attack_optimizer, batch_size):
    # switch to train mode
    attack_model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    len_data = min(len(labels), len(attack_labels))
    len_t = int(np.ceil(len_data/batch_size))
    for batch_ind in range(len_t):
        end_idx = min((batch_ind+1)*batch_size, len_data)

        outputs = infer_data[batch_ind*batch_size: end_idx].to(device,torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)
        outputs_non = attack_infer_data[batch_ind*batch_size: end_idx].to(device,torch.float)
        targets_attack = attack_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        comb_inputs = torch.cat((outputs,outputs_non))
        comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).to(device, torch.float)
        
        one_hot_tr = torch.zeros(comb_inputs.size()[0],comb_inputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).data, 1)
         
        attack_output = attack_model(comb_inputs, target_one_hot).view([-1])

        att_labels = torch.zeros((outputs.shape[0]+outputs_non.shape[0]))
        att_labels [:outputs.shape[0]] =1.0
        att_labels [outputs.shape[0]:] =0.0
        is_member_labels = att_labels.to(device, torch.float)
        
        loss_attack = attack_criterion(attack_output, is_member_labels)

        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy()>0.5), (is_member_labels.data.cpu().numpy()> 0.5)))        

        losses.update(loss_attack.item(), comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])
        
        # compute gradient and do SGD step
        attack_optimizer.zero_grad()
        loss_attack.backward()
        attack_optimizer.step()

    return (losses.avg, top1.avg)

def test_attack(infer_data, labels, attack_infer_data, attack_labels, attack_model, attack_criterion, batch_size):
    attack_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    len_data = min(len(labels), len(attack_labels))
    len_t = int(np.ceil(len_data/batch_size))
    member_prob = np.zeros(len_data)
    nonmember_prob = np.zeros(len_data)

    for batch_ind in range(len_t):
        end_idx = min(len_data, (batch_ind+1)*batch_size)

        outputs = infer_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)
        outputs_non = attack_infer_data[batch_ind*batch_size: end_idx].to(device,torch.float)
        targets_attack = attack_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        comb_inputs = torch.cat((outputs,outputs_non))
        comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).to(device,torch.float)      
        
        one_hot_tr = torch.zeros(comb_inputs.size()[0],comb_inputs.size()[1]).to(device, torch.float)
        target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).view([-1,1]).data,1)

        attack_output = attack_model(comb_inputs, target_one_hot).view([-1])

        att_labels = torch.zeros((outputs.shape[0]+outputs_non.size()[0]))
        att_labels [:outputs.shape[0]] =1.0
        att_labels [outputs.shape[0]:] =0.0

        is_member_labels = att_labels.to(device,torch.float)      
        
        loss = attack_criterion(attack_output, is_member_labels)
        
        member_prob[batch_ind*batch_size: end_idx]= attack_output.data.cpu().numpy()[: outputs.shape[0]]
        nonmember_prob[batch_ind*batch_size: end_idx]= attack_output.data.cpu().numpy()[outputs.shape[0]:]

        prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(is_member_labels.data.cpu().numpy()> 0.5)))
        losses.update(loss.item(), comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])
    
    return (losses.avg,top1.avg,member_prob,nonmember_prob)

def get_entropy(sample_predictions, sample_labels):
	#calculte entropy given prediction vectors(N,C) and labels(N), where N is the size of samples and C is the number of class.
	#lower is likely to be samples
	outputs = sample_predictions.copy()
	outputs[outputs<=0] = small_delta
	return np.sum(-outputs*np.log(outputs),axis=1)

def get_mentropy(sample_predictions, sample_labels):
	#calculate modified entropy given prediction vectors(N,C) and labels(N), where N is the size of samples and C is the number of class.
	#lower is likely to be samples	
	outputs = sample_predictions.copy()
	outputs[np.arange(len(sample_predictions)),sample_labels] = 1 - outputs[np.arange(len(sample_predictions)),sample_labels]
	outputs2 = 1-outputs
	outputs2[outputs==0] = small_delta
	return np.sum(-outputs*np.log(outputs2),axis=1)

def get_conf(sample_predictions, sample_labels):
	#higher is likely to be samples	
	return sample_predictions[np.arange(len(sample_predictions)), sample_labels]

def get_corrcect(sample_predictions, sample_labels):

    #higher is likely to be samples	
    return (np.argmax(sample_predictions,axis=1)==sample_labels).astype(int)

def threshold_based_inference_attack(train_member_stat,train_member_label,train_nonmember_stat,train_nonmember_label,test_member_stat,test_member_label,test_nonmember_stat,test_nonmember_label, num_class=100,per_class=True):
    """
    train_member_stat: member samples for finding threshold
    train_nonmember_stat: nonmember samples for finding threshold
    test_member_stat: member samples for MIA
    test_nonmember_stat: nonmember samples for evaluation MIA
    Note: Both stats are assumed to behave like confidence values, i.e., higher is better. Negate the values if it behaves in the opposite way, e.g., for xe-loss, lower is better
    """
    #global threshold 
    list_all = np.concatenate((train_member_stat, train_nonmember_stat))
    max_gap = 0
    thre_chosen_g = 0
    list_all.sort()
    for thre in list_all:
        ratio1 = np.sum(train_member_stat>=thre)
        ratio2 = len(train_nonmember_stat)-np.sum(train_nonmember_stat>=thre)
        if ratio1+ratio2 > max_gap:
            max_gap = ratio1+ratio2
            thre_chosen_g = thre
    #evaluate global threshold
    ratio1 = np.sum(test_member_stat>=thre_chosen_g)
    ratio2 = len(test_nonmember_stat)-np.sum(test_nonmember_stat>=thre_chosen_g)
    global_MIA_acc = (ratio1+ratio2)/(len(test_member_stat)+len(test_nonmember_stat))

    if per_class == True:
        #per-class threshold
        thre_chosen_class = np.zeros(num_class)
        for i in range(num_class):
            train_member_stat_class = train_member_stat[train_member_label==i]
            train_nonmember_stat_class = train_nonmember_stat[train_nonmember_label==i]
            list_all_class = np.concatenate((train_member_stat_class, train_nonmember_stat_class))
            max_gap = 0
            thre_chosen = 0
            list_all_class.sort()
            for thre in list_all_class:
                ratio1 = np.sum(train_member_stat_class>=thre)
                ratio2 = len(train_nonmember_stat_class)-np.sum(train_nonmember_stat_class>=thre)
                if ratio1+ratio2 > max_gap:
                    max_gap = ratio1+ratio2
                    thre_chosen = thre
            thre_chosen_class[i] = thre_chosen
        #evaluate per class threshold
        ratio1 = np.sum(test_member_stat>=thre_chosen_class[test_member_label])
        ratio2 = len(test_nonmember_stat) - np.sum(test_nonmember_stat>=thre_chosen_class[test_nonmember_label])
        class_MIA_acc = (ratio1+ratio2)/(len(test_member_stat)+len(test_nonmember_stat))
        return max(global_MIA_acc, class_MIA_acc), global_MIA_acc, thre_chosen_g, class_MIA_acc, thre_chosen_class
    else:
        return global_MIA_acc, thre_chosen_g

def nn_attack(train_member_pred,train_member_label,train_nonmember_pred,train_nonmember_label,test_member_pred,test_member_label,test_nonmember_pred,test_nonmember_label, attack_epochs=150, batch_size=512,num_class=100):
    """
    This assumes len(train_member_pred)==len(tran_nonmember_pred) and len(test_member_pred)==len(test_nonmember_pred)
    """
    test_member_pred_tensor = torch.from_numpy(test_member_pred).type(torch.FloatTensor)        
    test_member_label_tensor = torch.from_numpy(test_member_label).type(torch.LongTensor)
    test_nonmember_pred_tensor = torch.from_numpy(test_nonmember_pred).type(torch.FloatTensor)
    test_nonmember_label_tensor = torch.from_numpy(test_nonmember_label).type(torch.LongTensor)

    attack_model = InferenceAttack_HZ(num_class).to(device, torch.float)
    attack_criterion = nn.MSELoss().to(device, torch.float)
    attack_optimizer = optim.Adam(attack_model.parameters(),lr=0.0001)

    best_nn_acc= 0.0

    for epoch in range(0, attack_epochs):
        r= np.arange(len(train_member_pred))
        np.random.shuffle(r)
        train_member_pred = train_member_pred[r]
        train_member_label = train_member_label[r]
        r = np.arange(len(train_nonmember_pred))
        train_nonmember_pred = train_nonmember_pred[r]
        train_nonmember_label = train_nonmember_label[r]

        train_member_pred_tensor = torch.from_numpy(train_member_pred).type(torch.FloatTensor)        
        train_member_label_tensor = torch.from_numpy(train_member_label).type(torch.LongTensor)
        train_nonmember_pred_tensor = torch.from_numpy(train_nonmember_pred).type(torch.FloatTensor)
        train_nonmember_label_tensor = torch.from_numpy(train_nonmember_label).type(torch.LongTensor)

        train_loss, train_attack_acc = train_attack(train_member_pred_tensor, train_member_label_tensor, train_nonmember_pred_tensor, train_nonmember_label_tensor, attack_model, attack_criterion, attack_optimizer, batch_size)
        test_loss, test_attack_acc, mem, nonmem = test_attack(test_member_pred_tensor, test_member_label_tensor, test_nonmember_pred_tensor, test_nonmember_label_tensor, attack_model, attack_criterion, batch_size)

        is_best = test_attack_acc>best_nn_acc
        best_nn_acc = max(test_attack_acc, best_nn_acc)

    return best_nn_acc   

def system_attack(train_member_pred,train_member_label,test_member_pred,test_member_label,train_nonmember_pred,train_nonmember_label,test_nonmember_pred,test_nonmember_label, num_class=100, attack_epochs=150, batch_size=512):
    len1, len2 = min(len(train_member_pred), len(train_nonmember_pred)), min(len(test_member_pred), len(test_nonmember_pred))

    train_member_pred, train_member_label = train_member_pred[:len1], train_member_label[:len1]
    train_nonmember_pred, train_nonmember_label = train_nonmember_pred[:len1], train_nonmember_label[:len1]
    test_member_pred, test_member_label = test_member_pred[:len2], test_member_label[:len2]
    test_nonmember_pred, test_nonmember_label = test_nonmember_pred[:len2], test_nonmember_label[:len2]

    print("\n\nEvaluating direct single-query attacks :", len(train_member_pred), len(train_nonmember_pred), len(test_member_pred), len(test_nonmember_pred))
    print("batch_size", batch_size)	
    print(train_member_label[:20])
    print(test_member_label[:20])
    print(test_nonmember_label[:20])
    print(train_nonmember_label[:20])
    print ('classifier acc on attack training set: {:.4f}, {:.4f}.\nclassifier acc on attack test set:     {:.4f}, {:.4f}.'.format(np.mean(np.argmax(train_member_pred,axis=1)==train_member_label),np.mean(np.argmax(train_nonmember_pred,axis=1)==train_nonmember_label),np.mean(np.argmax(test_member_pred,axis=1)==test_member_label),np.mean(np.argmax(test_nonmember_pred,axis=1)==test_nonmember_label)))
    
    train_mem_stat = get_conf(train_member_pred, train_member_label)
    train_nonmem_stat = get_conf(train_nonmember_pred, train_nonmember_label)
    test_mem_stat = get_conf(test_member_pred, test_member_label)
    test_nonmem_stat = get_conf(test_nonmember_pred, test_nonmember_label)
    conf_acc,conf_acc_g,_,conf_acc_c,_ = threshold_based_inference_attack(train_mem_stat,train_member_label,train_nonmem_stat,train_nonmember_label,test_mem_stat,test_member_label,test_nonmem_stat,test_nonmember_label)

    train_mem_stat = -get_entropy(train_member_pred, train_member_label)
    train_nonmem_stat = -get_entropy(train_nonmember_pred, train_nonmember_label)
    test_mem_stat = -get_entropy(test_member_pred, test_member_label)
    test_nonmem_stat = -get_entropy(test_nonmember_pred, test_nonmember_label)
    entr_acc,entr_acc_g,_,entr_acc_c,_ = threshold_based_inference_attack(train_mem_stat,train_member_label,train_nonmem_stat,train_nonmember_label,test_mem_stat,test_member_label,test_nonmem_stat,test_nonmember_label)

    train_mem_stat = -get_mentropy(train_member_pred, train_member_label)
    train_nonmem_stat = -get_mentropy(train_nonmember_pred, train_nonmember_label)
    test_mem_stat = -get_mentropy(test_member_pred, test_member_label)
    test_nonmem_stat = -get_mentropy(test_nonmember_pred, test_nonmember_label)
    mentr_acc,mentr_acc_g,_,mentr_acc_c,_ = threshold_based_inference_attack(train_mem_stat,train_member_label,train_nonmem_stat,train_nonmember_label,test_mem_stat,test_member_label,test_nonmem_stat,test_nonmember_label)

    train_mem_stat = get_corrcect(train_member_pred, train_member_label)
    train_nonmem_stat = get_corrcect(train_nonmember_pred, train_nonmember_label)
    test_mem_stat = get_corrcect(test_member_pred, test_member_label)
    test_nonmem_stat = get_corrcect(test_nonmember_pred, test_nonmember_label)
    corr_acc,_ = threshold_based_inference_attack(train_mem_stat,train_member_label,train_nonmem_stat,train_nonmember_label,test_mem_stat,test_member_label,test_nonmem_stat,test_nonmember_label,per_class=False)

    nn_acc = nn_attack(train_member_pred,train_member_label,train_nonmember_pred,train_nonmember_label,test_member_pred,test_member_label,test_nonmember_pred,test_nonmember_label, attack_epochs=150, batch_size=512,num_class=100)

    print("Best direct single-query attack acc: {:.4f}. NN attack: {:.4f}. Correctness: {:.4f}. Global|Class:  Conf:{:.4f}|{:.4f}. Entr: {:.4f}|{:.4f}. Mentr: {:.4f}|{:.4f}".format(max(entr_acc,mentr_acc,conf_acc,corr_acc,nn_acc),nn_acc,corr_acc,conf_acc_g,conf_acc_c,entr_acc_g,entr_acc_c,mentr_acc_g,mentr_acc_c))

    return max(entr_acc,mentr_acc,conf_acc,corr_acc,nn_acc)