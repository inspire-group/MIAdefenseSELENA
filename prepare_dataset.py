import tarfile
import yaml
import os
from utils import mkdir_p
import numpy as np
import wget
import shutil

config_file = './env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']
##prepare
if os.path.exists(os.path.join(src_dir, 'memguard')):
    shutil.move(os.path.join(src_dir, 'memguard'), os.path.join(root_dir, 'memguard'))

if not os.path.exists(os.path.join(root_dir, 'tmp')):
    os.makedirs(os.path.join(root_dir, 'tmp'))
###assumeing two tar files dataset_purcahse.tgz and dataset_texas.tgz are saved in root_dir/tmp.
####prepare purchase dataset
if not os.path.isfile(os.path.join(root_dir, 'tmp', 'dataset_purchase.tgz')):
    print("Dowloading purchase dataset...")
    wget.download("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz", os.path.join(root_dir, 'tmp', 'dataset_purchase.tgz'))
    print('Dataset Dowloaded')

if not os.path.isfile(os.path.join(root_dir, 'tmp', 'dataset_texas.tgz')):
    print("Dowloading texas dataset...")
    wget.download("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz", os.path.join(root_dir, 'tmp', 'dataset_texas.tgz'))
    print('Dataset Dowloaded')

if not os.path.isfile(os.path.join(root_dir, 'tmp', 'cifar-100-python.tar.gz')):
    print("Dowloading cifar100 dataset...")
    wget.download("http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", os.path.join(root_dir, 'tmp', 'cifar-100-python.tar.gz'))
    print('Dataset Dowloaded')
if not os.path.exists(os.path.join(root_dir, 'cifar100', 'data')):
    print("Prepare CIFAR100 dataset")
    tar = tarfile.open(os.path.join(root_dir, 'tmp', 'cifar-100-python.tar.gz'))
    tar.extractall(path=os.path.join(root_dir, 'cifar100'))
    os.rename(os.path.join(root_dir, 'cifar100', 'cifar-100-python'), os.path.join(root_dir, 'cifar100', 'data'))

print("Prepare Purchase100 dataset")
tar = tarfile.open(os.path.join(root_dir, 'tmp', 'dataset_purchase.tgz'))
tar.extractall(path=os.path.join(root_dir, 'tmp'))
data_set =np.genfromtxt(os.path.join(root_dir, 'tmp', 'dataset_purchase'), delimiter=',')

X = data_set[:,1:].astype(np.float64)
Y = (data_set[:,0]).astype(np.int32)-1

DATASET_PATH = os.path.join(root_dir, 'purchase', 'data')
if not os.path.exists(DATASET_PATH):
    mkdir_p(DATASET_PATH)

np.save(os.path.join(DATASET_PATH, 'X.npy'), X)
np.save(os.path.join(DATASET_PATH,'Y.npy'), Y)

print("Prepare Texas100 dataset")
####prepare texas dataset
####prepare purchase dataset
tar = tarfile.open(os.path.join(root_dir, 'tmp', 'dataset_texas.tgz'))
tar.extractall(path=os.path.join(root_dir, 'tmp'))

data_set_features =np.genfromtxt(os.path.join(root_dir, 'tmp', 'texas/100/feats'), delimiter=',')
data_set_label =np.genfromtxt(os.path.join(root_dir, 'tmp', 'texas/100/labels'), delimiter=',')

X =data_set_features.astype(np.float64)
Y = data_set_label.astype(np.int32)-1

DATASET_PATH = os.path.join(root_dir, 'texas', 'data')
if not os.path.exists(DATASET_PATH):
    mkdir_p(DATASET_PATH)

######save dataset in numpy format as loading by genfromtxt takes several minutes when loading.
np.save(os.path.join(DATASET_PATH, 'feats.npy'), X)
np.save(os.path.join(DATASET_PATH, 'labels.npy'), Y)
