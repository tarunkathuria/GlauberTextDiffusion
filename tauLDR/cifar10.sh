#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --output=./slurm-outputs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=25g
#SBATCH -t 0
####SBATCH -x tir-0-9,tir-0-7,tir-0-13,tir-0-15,tir-0-17,tir-0-19,tir-0-11,tir-0-32,tir-0-36,tir-1-13,tir-0-3,tir-1-11,tir-1-28
#####SBATCH -x tir-0-36,tir-1-28,tir-1-18,tir-1-23,tir-1-13
########SBATCH -x tir-0-19,tir-0-36,tir-0-32,tir-0-17,tir-1-28,tir-1-11,tir-0-11
#SBATCH -x tir-0-3,tir-0-11,tir-0-32,tir-0-36

set -x  # echo commands to stdout
set -e  # exit on error
#module load cuda-8.0 cudnn-8.0-5.1
#export CUDE_VISIBLE_DEVICES=2,1
# echo "$@"
# sh "$@"
# echo "ok done"
export HF_DATASETS_CACHE="/projects/tir6/general/sachink/huggingface"
# conda activate 2022
source /projects/tir1/users/sachink/data/anaconda3/bin/activate 2022
# echo "ok done"
pwd

python train.py cifar10
