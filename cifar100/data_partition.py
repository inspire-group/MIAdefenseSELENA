import os
import random
import pickle
import numpy as np
import yaml

config_file = './../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    path_dir = os.path.join(root_dir, 'cifar100')
    DATASET_PATH = os.path.join(path_dir, 'data')

    tmp = unpickle(os.path.join(DATASET_PATH, 'cifar-100-python', 'train'))
    X = np.zeros((tmp[b'data'].shape[0], 3, 32, 32))
    X[:,0,:,:] = np.array(tmp[b'data'])[:,:1024].reshape((-1, 32,32))
    X[:,1,:,:] = np.array(tmp[b'data'])[:,1024:1024*2].reshape((-1, 32,32))
    X[:,2,:,:] = np.array(tmp[b'data'])[:,1024*2:].reshape((-1, 32,32))
    Y = np.array(tmp[b'fine_labels'])

    len_train =len(Y)
    if os.path.exists(os.path.join(DATASET_PATH,'random_r_cifar100')):
        r=pickle.load(open(os.path.join(DATASET_PATH,'random_r_cifar100'),'rb'))
    else:
        r = np.arange(len_train)
        np.random.shuffle(r)
        pickle.dump(r,open(os.path.join(DATASET_PATH,'random_r_cifar100'),'wb'))

    X=X[r]
    Y=Y[r]

    np.random.seed(0)

    r = np.arange(len_train)
    np.random.shuffle(r)

    X = X[r]
    Y = Y[r]

    train_data = X
    train_label = Y

    np.random.seed(1000)
    r = np.arange(len(train_data))
    np.random.shuffle(r)
    train_data_tr_attack = train_data[r[:int(0.5*len(r))]]
    train_label_tr_attack = train_label[r[:int(0.5*len(r))]]

    train_data_te_attack = train_data[r[int(0.5*len(r)):]]
    train_label_te_attack = train_label[r[int(0.5*len(r)):]]

    tmp = unpickle(os.path.join(DATASET_PATH, 'cifar-100-python', 'test'))
    test_X = np.zeros((tmp[b'data'].shape[0], 3, 32, 32))
    test_X[:,0,:,:] = np.array(tmp[b'data'])[:,:1024].reshape((-1, 32,32))
    test_X[:,1,:,:] = np.array(tmp[b'data'])[:,1024:1024*2].reshape((-1, 32,32))
    test_X[:,2,:,:] = np.array(tmp[b'data'])[:,1024*2:].reshape((-1, 32,32))
    test_Y = np.array(tmp[b'fine_labels'])

    ref_ratio = 0.5
    np.random.seed(2000)

    all_test_data = test_X.copy()
    all_test_label = test_Y.copy()

    r = np.arange(len(test_X))
    np.random.shuffle(r)
    ref_data = test_X[r[int((ref_ratio)*len(test_Y)):]]
    test_data = test_X[r[:int(ref_ratio*len(test_Y))]]

    ref_label = test_Y[r[int((ref_ratio)*len(test_Y)):]]
    test_label = test_Y[r[:int(ref_ratio*len(test_Y))]]

    print(train_label_tr_attack[:20])
    print(train_label_te_attack[:20])
    print(test_label[:20])
    print(ref_label[:20])

    path2 = os.path.join(DATASET_PATH, 'partition')
    if not os.path.isdir(path2):
        mkdir_p(path2)

    np.save(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'), train_data_tr_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'), train_label_tr_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'), train_data_te_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'), train_label_te_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_data.npy'), train_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_label.npy'), train_label)
    np.save(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'), ref_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'), ref_label)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'), test_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'), test_label)
    np.save(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'), all_test_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'), all_test_label)


if __name__ == '__main__':
    main()
