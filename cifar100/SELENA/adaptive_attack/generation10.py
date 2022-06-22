import os
import random
import pickle
import numpy as np
import yaml
import argparse

config_file = './../../../env.yml'
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
    parser = argparse.ArgumentParser(description = 'K and L for SELENA')
    parser.add_argument('--K', type = int, default = 25, help = 'total sub-models in split-ai')
    parser.add_argument('--L', type = int, default = 10, help = 'non_model for each sample in split-ai')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    path_dir = os.path.join(root_dir, 'cifar100')
    DATASET_PATH = os.path.join(path_dir, 'data')

    split_model = args.K
    non_model = args.L
    split_name = str(split_model) + '_' + str(non_model)
    print("Generate L partition for K/L: {:d}/{:d}".format(split_model, non_model))

    path2 = os.path.join(DATASET_PATH, 'partition', 'K_L', split_name, 'attacker')

    if not os.path.isdir(path2):
        mkdir_p(path2)

    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))

    tr_L_models = np.zeros((train_label_tr_attack.shape[0], 1+non_model))
    te_L_models = np.zeros((train_label_te_attack.shape[0], 1+non_model))

    #attacker specify a random seed
    np.random.seed(2000)
    for i in range(len(tr_L_models)):
        tr_L_models[i, 0] = train_label_tr_attack[i]
        tmp = np.arange(split_model)
        np.random.shuffle(tmp)
        tr_L_models[i, 1:] = tmp[:non_model]        
    tr_L_models = tr_L_models.astype(np.int32)
    for i in range(len(te_L_models)):
        te_L_models[i, 0] = train_label_te_attack[i]
        tmp = np.arange(split_model)
        np.random.shuffle(tmp)
        te_L_models[i, 1:] = tmp[:non_model]
    te_L_models = te_L_models.astype(np.int32)
    np.save(os.path.join(path2, 'tr_label.npy'), tr_L_models)
    np.save(os.path.join(path2, 'te_label.npy'), te_L_models)

if __name__ == '__main__':
    main()