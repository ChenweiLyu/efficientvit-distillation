#!/bin/bash

#SBATCH -p a100-gpu-shared
#SBATCH -t 4-00:00:00
#SBATCH -J train
#SBATCH --gres=gpu:1
#SBATCH --output=/storage2/datasets/chenwei/code/efficientvit/job_log/%A.out
 
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate di

cd /storage2/datasets/chenwei/code/efficientvit
python distillation.py