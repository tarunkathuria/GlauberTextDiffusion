#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=25G

source /projects/tir1/users/sachink/data/anaconda3/bin/activate 2022

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
