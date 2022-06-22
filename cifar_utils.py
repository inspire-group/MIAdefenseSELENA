from PIL import Image
import numpy as np

import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


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

class DistillCifardata(data.Dataset):
    def __init__(self, data, confs, conf_labels, labels, transform):
        self.data = data
        self.transform = transform
        self.confs = confs
        self.conf_labels = conf_labels
        self.labels = labels

    def __getitem__(self, index):
        img =  Image.fromarray((self.data[index].transpose(1,2,0).astype(np.uint8)))
        label = self.labels[index]
        img = self.transform(img)
        conf = self.confs[index]
        conf_label = self.conf_labels[index]

        return img, conf, conf_label, label

    def __len__(self):
        return len(self.labels)

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class ModelwNorm(nn.Module):
    def __init__(self, model):
        super(ModelwNorm, self).__init__()
        self.model = model
        self.mean = torch.tensor([0.507, 0.487, 0.441]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.267, 0.256, 0.276]).view(1, 3, 1, 1)
    def forward(self, x):
        m, s = self.mean.to(x.device), self.std.to(x.device)
        return self.model((x-m)/s)

