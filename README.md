# DNLP SS23 Final Project - Multitask BERT
Team: G04	Semantic Foxes

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!) 

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

## Part 1: minBERT
Getting access to GPU

```
srun --pty -p grete:interactive  -G V100:1 /bin/bash
```

Commands for training the model
```
python classifier.py --option finetune --lr 1e-5 --hidden_dropout_prob=0.1 --local_files_only --use_gpu 
```
```
python classifier.py --option pretrain --lr 1e-3 --hidden_dropout_prob=0.1 --local_files_only --use_gpu 
```
