#!/bin/bash
#SBATCH --job-name=lt_tecoa

#SBATCH -p gpu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# 1) 原始 CLIP baseline（不微调）
python robust_finetune_cifar100_lt_experiments.py \
  --no_finetune \
  --dataset cifar100 \
  --batch_size 16 \
  --test_eps 2 --test_numsteps 5 --test_stepsize 1\
  --name ori_CLIP_baseline_nofinetunign_version_1

# # 2) 标准 CIFAR100 对抗微调
# python robust_finetune_cifar100_lt_experiments.py \
#   --dataset cifar100 \
#   --epochs 10 \
#   --batch_size 16 \
#   --train_eps 2 --train_numsteps 5 --train_stepsize 1 \
#   --test_eps 2 --test_numsteps 5 --test_stepsize 1

# # 3) CIFAR100-LT-100 对抗微调
# python robust_finetune_cifar100_lt_experiments.py \
#   --dataset cifar100_lt \
#   --imbalance_factor 100 \
#   --lt_seed 0 \
#   --epochs 10 \
#   --batch_size 16 \
#   --train_eps 2 --train_numsteps 5 --train_stepsize 1 \
#   --test_eps 2 --test_numsteps 5 --test_stepsize 1

# # 4) same-budget balanced 对照组
# python robust_finetune_cifar100_lt_experiments.py \
#   --dataset cifar100_balanced_subset \
#   --imbalance_factor 100 \
#   --lt_seed 0 \
#   --epochs 10 \
#   --batch_size 16 \
#   --train_eps 2 --train_numsteps 5 --train_stepsize 1 \
#   --test_eps 2 --test_numsteps 5 --test_stepsize 1