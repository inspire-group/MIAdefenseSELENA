import tarfile
import yaml
import os
from utils import mkdir_p
import numpy as np

config_file = './env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']


###assumeing two tar files dataset_purcahse.tgz and dataset_texas.tgz are saved in root_dir/tmp.
####prepare purchase dataset
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