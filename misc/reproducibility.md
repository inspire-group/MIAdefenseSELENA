# Overview
This document provides a detailed guide to reproduce experimental results in the main body of our MIAdefenseSELENA paper, i.e. Table 2.

## Files
```shell
├── MIAdefenseSELENA
|    ├── env.yml      # specify root_dir and src_dir (ending with MIAdefenseSELENA)
|    ├── requirement.txt
|    ├── utils.py
|    ├── cifar_utils.py
|    ├── attack
|    |    ├── dsq_attack.py               # direct single-query attacks on purchase100/texas100/cifar100
|    |    ├── binary_flip_noise_attack.py # label-only attacks on purchase100/texas100
|    |	  ├── Aug_Attack.py               # augmenation attacks (data augmentation attacks, label-only) on cifar100
|    |    ├── CW_Attack.py                # cw attacks (boundary attacks, label-only) on cifar100
|    | 	  └── adaptive_attack.py          # adaptive attacks for SELENA on purchase100/texas100/cifar100 on
|    ├── models
|    |    ├── purchase.py # model for Purchase100
|    |	  ├── texas.py    # model for Texas100
|    |	  └── resnet.py   # model for CIFAR100   
|    ├── purchase
|    |    ├── data_partition.py # generate * npy files for member/nonmember sets to train/eval MIA attacks 
|    |    ├── Undefend
|    |    |    ├── train.py      # train the undefended model
|    |	  |    └── eval.py       # eval the undefended model via direct single-query attacks and label-only attacks
|    |	  ├── MemGuard
|    |	  |    ├── prepare_for_memguard.py  # save the predictions and logits as inputs for memguard
|    |	  |    ├── memguard_run.py          # memguard defense, output the perturbed predictions
|    |	  |    └── eval_memguard.py         # eval the memguard perturbed predictions via direct single-query attacks
|    |	  ├── AdvReg
|    |	  |    ├── train.py     # train the model via adversarial training
|    |	  |    └── eval.py      # eval the AdvReg model via direct single-query attacks and label-only attacks
|    |	  └── SELENA
|    |	       ├── generation10.py # generate non_model indices for defender's Split-AI
|    |	       ├── Split_AI
|    |	       |    ├── train.py   # train the Split-AI model
|    |	       |    └── eval.py    # eval the Split-AI model via direct single-query attacks (expect ~50\% accuracy) 
|    |	       ├── Distillation
|    |	       |    ├── train.py   # train a new model via distillation from Split-AI 
|    |	       |    └── eval.py    # eval the distillation model (i.e., the final output protected model) via direct single-query attacks and label-only attacks
|    |	       └── adaptive_attack
|    |	            ├── generation10.py # generate non_model indices for attacker's shadow Split-AI
|    |	            ├── train.py        # train the attacker's shadow Split-AI
|    |	            └── eval.py         # evaluate the distillation model (i.e., the final output protected model) via adaptive attacks
|    ├── texas    # The file structure in this folder is the same as in MIAdefenseSELENA/purchase
|    ├── cifar100
|    |    ├── data_partition.py  # generate * npy files for member/nonmember sets to train/eval MIA attacks 
|    |	  ├── Undefend
|    |	  |    ├── train.py      # train the undefended model
|    |	  |    └── eval.py       # eval the undefended model via direct single-query attacks
|    |	  |    ├── eval_aug.py   # eval the undefended model via augmenation attacks (data augmentation attacks, label-only)
|    |	  |    └── eval_cw.py    # eval the undefended model via cw attacks (boundary attacks, label-only)
|    |    ├── MemGuard
|    |	  |    ├── prepare_for_memguard.py  # save the predictions and logits as inputs for memguard
|    |	  |    ├── memguard_run.py          # memguard defense, output the perturbed predictions
|    |	  |    └── eval_memguard.py         # eval the memguard perturbed predictions via direct single-query attacks
|    |    ├── AdvReg
|    |	  |    ├── train.py      # train the model via adversarial training
|    |    |    ├── eval.py       # eval the AdvReg model via direct single-query attacks
|    |    |    ├── eval_aug.py   # eval the AdvReg model via augmenation attacks (data augmentation attacks, label-only)
|    |	  |    └── eval_cw.py    # eval the AdvReg model via cw attacks (boundary attacks, label-only)
|    |	  └── SELENA
|    |	       ├── generation10.py # generate non_model indices for defender's Split-AI
|    |	       ├── Split_AI
|    |	       |    ├── train.py   # train the Split-AI model
|    |	       |    └── eval.py    # eval the Split-AI model via direct single-query attacks (expect ~50\% accuracy) 
|    |	       ├── Distillation
|    |	       |    ├── train.py     # train a new model via distillation from Split-AI 
|    |	       |    ├── eval.py      # eval the distillation model (i.e., the final output protected model) via direct single-query attacks
|    |	       |    ├── eval_aug.py  # eval the distillation model (i.e., the final output protected model) via augmenation attacks (data augmentation attacks, label-only)
|    |	       |    └── eval_cw.py   # eval the distillation model (i.e., the final output protected model) via cw attacks (boundary attacks, label-only)
|    |	       └── adaptive_attack
|    |	            ├── generation10.py # generate non_model indices for attacker's shadow Split-AI
|    |	            ├── train.py        # train the attacker's shadow Split-AI
|    |	            └── eval.py         # evaluate the distillation model (i.e., the final output protected model) via adaptive attacks
└── MIA_root_dir
     ├── memguard    # attack models for the optimization process of memguard
     |    ├── purchase_MIA_model.h5
     |    ├── texas_MIA_model.h5
     |	  └── cifar100_MIA_model.h5
     ├── purchase
     |	  ├── data
     |	  |    ├── random_r_purchase100
     |    |    ├── X.npy
     |    |    ├── Y.npy
     |    |    ├── memguard
     |    |    |    ├── defense_results # the output of memguard
     |    |    |    └── prediction      # predictions and logits for the inputs of memguard
     |	  |    └── partition
     |	  |         ├──*.npy  # npy files for member/nonmember sets to train/eval MIA attacks 
     |	  |         └── K_L
     |    |             └── 25_10
     |	  |	            ├── defender  # non_model indices for defender's Split-AI
     |	  |                 └── attacker  # non_model indices for attacker's shadow Split-AI
     |	  └── checkpoints
     |	      ├── undefend   # model of undefended model
     |	      ├── AdvReg     # model of adversarial training
     |	      └── K_L
     |	          └── 25_10
     |                ├── split_ai # models of defender's Split-AI
     |                ├── selena   # model of defender's Distillation, i.e., the final output model
     |	      	      └── shadow   # models of attacker's shadow Split-AI
     ├── texas
     |    ├── data
     |    |    ├── random_r_texas100
     |    |    ├── feats.npy
     |    |    ├── labels.npy
     |    |    └── remaining files # remaining files have the same structure as MIA_root_dir/purchase/data (and will be created after running the corresponding codes)
     |	  └── checkpoints # files in this folder have the same structure as MIA_root_dir/purchase/checkpoints (and will be created after running the corresponding codes)
     └──  cifar100
      	  ├── data
      	  |    ├── random_r_cifar100
          |    ├── cifar-100-python
          |    └── remaining files # remaining files have the same structure as MIA_root_dir/purchase/data (and will be created after running the corresponding codes)
          └── checkpoints # files in this folder have the same structure as MIA_root_dir/purchase/checkpoints (and will be created after running the corresponding codes)      
```

Specify you ```root_dir``` and ```src_dir``` in env.yml

## Requirements
The code is tested with python 3.8.5, PyTorch 1.11.0 (for most of the experiments) and TensorFlow-2.9.1 (for MemGuard). The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`.

## Datasets
- Purchase100 [[downloading link](https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz)] (needs to be converted to X.npy and Y.npy and save to MIA_root_dir/purchase/data)
- Texas100 [[downloading link](https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz)] (needs to be converted to feats.npy and labels.npy and save to MIA_root_dir/texas/data)
- CIFAR100 [[downloading link](http://www.cs.toronto.edu/~kriz/cifar.html)] (download the cifar100 dataset cifar-100-python.tar.gz and untar it to MIA_root_dir/cifar100/data)
- You may refer [Adversarial Regularization](https://github.com/SPIN-UMass/ML-Privacy-Regulization) on how to load Purchase100/Texas100 . After preparing corresponding files following the files structure, and specifying your env.yml, you can proceed to usage.

## Usage
You may refer file structures and comments in [`Files`](./reproducibility.md#files).

For each dataset $datasetname (purchase/texas/cifar100)

First of all, generate the npy files for member/nonmember sets to train/eval MIA attacks 
```
cd $datasetname
python data_partition.py
```

### SELENA
First prepare the non_model indices for defenfer's Split-AI
```
cd $datasetname/SELENA/
python generate10.py
```
Then train the models for Split-AI
```
cd $datasetname/SELENA/Split_AI
python train.py
```
Next train the model from self-distillation from Split-AI
```
cd $datasetname/SELENA/Distillation
python train.py
```
To evaluate the final protected model (models via self-distillation from Split-AI) through direct single-query attacks and label-only attacks.
```
python eval.py
``` 
For purchase100/texas100, eval.py includes direct single-query attacks and label-only attacks.
For cifar100, eval.py includes direct single-query attacks. eval_aug.py includes data augmentation attacks (label-only). eval_cw.py includes boundary distance attacks (label-only). 

To evaluate the final protected model (models via self-distillation from Split-AI) through adaptive attacks.

First prepare the non_model indices for attacker's shadow Split-AI
```
cd $datasetname/SELENA/adaptive_attack
python generate10.py
```
Then train attacker's shadow Split-AI
```
python train.py
```
Next evaluate the final protected model (models via self-distillation from Split-AI) through adaptive attacks.
```
python eval.py
```

### undefended model:
To train a undefended model
```
cd $datasetname/undefend
python train.py
```
To evaluate it via MIA attacks
```
python eval.py
```
For purchase100/texas100, eval.py includes direct single-query attacks and label-only attacks.
For cifar100, eval.py includes direct single-query attacks. eval_aug.py includes data augmentation attacks (label-only). eval_cw.py includes boundary distance attacks (label-only). 
### MemGuard
First train the undefended model following [```undefended-model```](./reproducibility.md#undefended-model)

Download NN MIA attack model via Google drive [link](https://drive.google.com/drive/folders/1Lu8OO1bJZRNrO8o44rSPIZmUlNoyUZNn?usp=sharing) and save to MIA_root_dir/memguard.

To prepare the logits and predictions as inputs to MemGuard
 ```
cd $datasetname/MemGuard
python prepare_for_memguard.py
```
To run MemGuard code
 ```
python memguard_run.py
```
To evaluate MemGuard via direct single-query attacks
 ```
python eval_memguard.py
```
The label-only attack for MemGuard is the same as that for undefended model, no extra experiment performed.

### Adversarial Regularization
To train a model via Adversarial Regularization
```
cd $datasetname/AdvReg
python train.py
```
To evaluate it via MIA attacks
```
python eval.py
```
For purchase100/texas100, eval.py includes direct single-query attacks and label-only attacks.
For cifar100, eval.py includes direct single-query attacks. eval_aug.py includes data augmentation attacks (label-only). eval_cw.py includes boundary distance attacks (label-only). 
