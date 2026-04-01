#!/bin/bash
#SBATCH --job-name=lt_tecoa

#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

module load anaconda3/2023.09
source activate tecoa
cd /scratch/hpc/07/zhang303/tecoa


# 1) 原始 CLIP baseline（不微调）
# srun python at_lt_test.py \
#   --no_finetune \
#   --num_workers 8 \
#   --dataset cifar100 \
#   --batch_size 16 \
#   --test_eps 2 --test_numsteps 5 --test_stepsize 1\
#   --name ori_CLIP_baseline_nofinetuning_version_1.1
# 20687282,

# 2) 标准 CIFAR100 对抗微调
# srun python at_lt_test.py \
#   --dataset cifar100 \
#   --num_workers 8 \
#   --epochs 10 \
#   --batch_size 16 \
#   --train_eps 2 --train_numsteps 5 --train_stepsize 1 \
#   --test_eps 2 --test_numsteps 5 --test_stepsize 1\
#   --name standard_CLIP_finetuning_version_1
# #   20692951



# 3) CIFAR100-LT-100 对抗微调
srun python at_lt_test.py \
  --dataset cifar100_lt \
  --num_workers 8 \
  --imbalance_factor 100 \
  --lt_seed 0 \
  --epochs 10 \
  --batch_size 16 \
  --train_eps 2 --train_numsteps 5 --train_stepsize 1 \
  --test_eps 2 --test_numsteps 5 --test_stepsize 1\
  --name CLIP_lt_finetuning_version_1
#   20688444, 20713700

# # 4) same-budget balanced 对照组
# srun python at_lt_test.py \
#   --dataset cifar100_balanced_subset \
#   --num_workers 8 \
#   --imbalance_factor 100 \
#   --lt_seed 0 \
#   --epochs 10 \
#   --batch_size 16 \
#   --train_eps 2 --train_numsteps 5 --train_stepsize 1 \
#   --test_eps 2 --test_numsteps 5 --test_stepsize 1\
#   --name CLIP_same-budget_balanced_finetuning_version_1
# #   20688447,