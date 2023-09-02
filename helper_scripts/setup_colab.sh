#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
export PYTHONPATH=""
./Miniconda3-latest-Linux-x86_64.sh -b
export PATH="$HOME/miniconda3/bin:$PATH"
conda init
source ~/.bashrc
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
conda env create -f dnlp.yml -y
conda activate dnlp
export WANDB_ENTITY=semantic_foxes
export WANDB_API_KEY=22afe59779abd82e1b93402e3debae92be971453
