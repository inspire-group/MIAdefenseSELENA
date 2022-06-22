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

def main():
    path_dir = os.path.join(root_dir, 'purchase')
    DATASET_PATH = os.path.join(path_dir, 'data')

    DATASET_FEATURES = os.path.join(DATASET_PATH,'X.npy')
    DATASET_LABELS = os.path.join(DATASET_PATH,'Y.npy')

    X = np.load(DATASET_FEATURES)
    Y = np.load(DATASET_LABELS)

    len_train =len(Y)
    if os.path.exists(os.path.join(DATASET_PATH,'random_r_purchase100')):
        r=pickle.load(open(os.path.join(DATASET_PATH,'random_r_purchase100'),'rb'))
    else:
        r = np.arange(len_train)
        np.random.shuffle(r)
        pickle.dump(r,open(os.path.join(DATASET_PATH,'random_r_purchase100'),'wb'))

    np.random.seed(0)
    X=X[r]
    Y=Y[r]
    r = np.arange(len_train)
    np.random.shuffle(r)
    X = X[r]
    Y = Y[r]

    train_classifier_ratio, train_attack_ratio = 0.1, 0.15
    train_data = X[:int(train_classifier_ratio*len_train)]
    ref_data = X[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    all_test_data = X[int(train_classifier_ratio*len_train):]

    train_label = Y[:int(train_classifier_ratio*len_train)]
    ref_label = Y[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    all_test_label = Y[int(train_classifier_ratio*len_train):]

    np.random.seed(1000)
    r = np.arange(len(train_data))
    np.random.shuffle(r)
    train_data_tr_attack = train_data[r[:int(0.5*len(r))]]
    train_label_tr_attack = train_label[r[:int(0.5*len(r))]]

    train_data_te_attack = train_data[r[int(0.5*len(r)):]]
    train_label_te_attack = train_label[r[int(0.5*len(r)):]]

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