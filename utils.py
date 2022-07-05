import os
import numpy as np
import torch.utils.data as data

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

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

def print_acc_conf(infer_np, test_labels):
    cal = np.zeros(19)
    calacc = np.zeros(19)
    conf_metric = np.max(infer_np, axis = 1)
    conf_metric_ind = np.argmax(infer_np, axis = 1)
    conf_avg = np.mean(conf_metric)
    acc_avg = np.mean(conf_metric_ind==test_labels)
    print("Total data: {:d}. Average acc: {:.4f}. Average confidence: {:.4f}.".format(len(infer_np), acc_avg, conf_avg))

    return acc_avg, conf_avg

class binarydata(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        img =  self.data[index]
        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.labels)    