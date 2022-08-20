# Overview
This document provides a detailed guide to reproduce experimental results in the main body of our MIAdefenseSELENA paper, i.e. Table 2.

## Files
```shell
├── MIAdefenseSELENA
|    ├── memguard      # pretrained NN MIA attack model for MemGuard
|    ├── env.yml      # specify root_dir and src_dir (ending with MIAdefenseSELENA)
|    ├── requirement.txt
|    ├── utils.py
|    ├── cifar_utils.py
|    ├── prepare_dataset.py     # prepare purhcase100 (X,npy, Y.npy) and texas100 (feats.npy, labels.npy)
|    ├── early_stopping.py      # load checkpoints saved in each epoch during undefended training, launch direct single-query attacks and plot Figure 4 in the paper.
|    ├── attack
|    |    ├── dsq_attack.py               # direct single-query attacks on purchase100/texas100/cifar100
|    |    ├── binary_flip_noise_attack.py # label-only attacks on purchase100/texas100
|    |	  ├── Aug_Attack.py               # augmenation attacks (data augmentation attacks, label-only) on cifar100
|    |    ├── CW_Attack.py                # cw attacks (boundary attacks, label-only) on cifar100
|    | 	  └── adaptive_attack.py          # adaptive attacks for SELENA on purchase100/texas100/cifar100
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

## Getting Started
Before running code, you may need to follow these three steps to prepare:

- Specifying you ```root_dir``` and ```src_dir``` in env.yml. ```root_dir``` is the root directory to save the data and checkpoints (corresponds to MIA_root_dir in [`Files`](./README.md#files)). ```src_dir``` is the root directory of the sourcecode (should endwith this repository name ```MIAdefenseSELENA```).

- Installing required packages. The code is tested with python 3.8.5, PyTorch 1.11.0 (for most of the experiments) and TensorFlow-2.9.1 (for MemGuard). The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`.

- Preparing Datasets and pretrained NN MIA attack model for memguard. We use three datasets: Purchase100 [[link](https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz)], Texas100 [[link](https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz)], CIFAR100 [[link](http://www.cs.toronto.edu/~kriz/cifar.html)]. You can prepare all three datasets by simply running the following command (this command will also move the pretrained NN MIA attack model for MemGuard to the assumed file path MIA_root_dir/memguard):
```
python prepare_datatset.py
```

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
python generation10.py
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
python generation10.py
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

### Figure 4 in Paper:
First train the undefended model following [```undefended-model```](./reproducibility.md#undefended-model). The script saves checkpoints for each epoch during the undefended model training.

[early_stopping.py](./../early_stopping.py) will load checkpoints saved in each epoch during undefended training, launch direct single-query attacks and plot Figure 4 in the paper.

```
python early_stopping.py
```
Please note that SELENA test accuracy and SELENA MIA accuracy are from Table 2.
