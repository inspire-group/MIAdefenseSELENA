import os
import random
import numpy as np
import argparse
import yaml
import sys

config_file = './../../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
from utils import mkdir_p

def main():
    parser = argparse.ArgumentParser(description = 'root dir')
    parser.add_argument('--K', type = int, default = 25, help = 'total sub-models in split-ai')
    parser.add_argument('--L', type = int, default = 10, help = 'non_model for each sample in split-ai')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    path_dir = os.path.join(root_dir, 'texas')
    DATASET_PATH = os.path.join(path_dir, 'data')

    split_model = args.K
    non_model = args.L
    split_name = str(split_model) + '_' + str(non_model)
    print("Generate L partition for K/L: {:d}/{:d}".format(split_model, non_model))

    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))

    tr_L_models = np.zeros((train_data_tr_attack.shape[0], train_data_tr_attack.shape[1]+non_model))
    te_L_models = np.zeros((train_data_te_attack.shape[0], train_data_te_attack.shape[1]+non_model))

    #set your own random seed
    np.random.seed(1000)
    for i in range(len(tr_L_models)):
        tr_L_models[i, :-non_model] = train_data_tr_attack[i, :]
        tmp = np.arange(split_model)
        np.random.shuffle(tmp)
        tr_L_models[i, -non_model:] = tmp[:non_model]

    for i in range(len(te_L_models)):
        te_L_models[i, :-non_model] = train_data_te_attack[i, :]
        tmp = np.arange(split_model)
        np.random.shuffle(tmp)
        te_L_models[i, -non_model:] = tmp[:non_model]

    path1 = os.path.join(DATASET_PATH, 'partition', 'K_L', split_name, 'defender')

    if not os.path.isdir(path1):
        mkdir_p(path1)

    np.save(os.path.join(path1, 'tr_data.npy'), tr_L_models)
    np.save(os.path.join(path1, 'te_data.npy'), te_L_models)

if __name__ == '__main__':
    main()