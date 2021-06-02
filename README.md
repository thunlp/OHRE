# OHRE

Dataset and code for NAACL 2021 paper: Open Hierarchical Relation Extraction.



## Required packages:

```
torch==1.3.0.post2
torchsummary==1.5.1
metrics==0.3.3
numpy==1.16.2
torchvision==0.4.2
scikit-learn==0.20.3
python-louvain==0.13
matplotlib==3.0.3
```

OR install with:

> pip install -r requirements.txt



## Data

FewRel data and the preprocessing code are under the directory `./data`.

NYT-FB data is from [Discrete-State Variational Autoencoders for Joint Discovery and Factorization of Relations](https://www.aclweb.org/anthology/Q16-1017.pdf)



## Run:

#### OpenRE Setting(e.g. on FewRel Hierarchy)

- **with dynamic triplet loss (default):**

`python train_OHRE.py --dataset ori --gpu 0`

- **with virtual adversarial training:**

`python train_OHRE.py --dataset ori --gpu 0 --trainset_loss_type triplet_v_adv`

#### Hierarchy Expansion Setting (e.g. on FewRel Hierarchy)

- **with dynamic triplet loss (default):**

`python train_OHRE_hierarchy_eval_louvain.py --dataset ori --gpu 0`

- **with virtual adversarial training:**

`python train_OHRE_hierarchy_eval_louvain.py --dataset ori --gpu 0 --trainset_loss_type triplet_v_adv ` 



## Cite

If you use the dataset or the code, please cite this paper:

```
@inproceedings{zhang2021Open,
  title={Open Hierarchical Relation Extraction},
  author={Kai Zhang, Yuan Yao, Ruobing Xie, Xu Han, Zhiyuan Liu, Fen Lin, Leyu Lin, Maosong Sun},
  booktitle={Proceedings of NAACL 2021},
  year={2021}
}
```




## Question

If you have any questions, feel free to contact `drogozhang@gmail.com`.
