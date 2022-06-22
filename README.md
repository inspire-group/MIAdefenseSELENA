# MIAdefenseSELENA
By Xinyu Tang, Saeed Mahloujifar, Liwei Song, Virat Shejwalkar, Milad Nasr, Amir Houmansadr, Prateek Mittal

Code for "Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture" in USENIX Security 2022.

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
- You may refer file structures and comments in [`Files`](./README.md#files).
- See [`misc/reproducibility.md`](./misc/reproducibility.md) for instructions to reproduce all results in the main body of paper.

## Notes 
Some variable names may not be consistent (and kind of confusing). The following variable names are equivalent.
- train_member_label/know_train_label/train_label_tr: for member sets to train MIA model.
- test_member_label/unknow_train_label_train_label_te: for member sets to eval MIA model.
- train_nonmember_label/ref_label/attack_label: for nonmember sets to train MIA model.
- test_nonmember_label/test_label: for nonmember sets to eval MIA model.


## Reference Repository
* Adversarial Regularization: https://github.com/SPIN-UMass/ML-Privacy-Regulization
* MemGuard: https://github.com/jinyuan-jia/MemGuard
* Systematic direct single-query attacks: https://github.com/inspire-group/membership-inference-evaluation
* Label-only attacks: https://github.com/cchoquette/membership-inference


## Citations

If you find our work useful in your research, please consider citing:

```tex
@inproceedings{tang2022miadefenseselena,
  title={Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture},
  author={Tang, Xinyu and Mahloujifar, Saeed and Song, Liwei and Shejwalkar, Virat and Nasr, Milad and Houmansadr, Amir and Mittal, Prateek},
  booktitle = {31st {USENIX} Security Symposium ({USENIX} Security)},
  year={2022}
}
```