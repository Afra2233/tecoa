#!/bin/bash
#SBATCH --job-name=TECOA

#SBATCH -p gpu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

module load anaconda3/2023.09
source activate tecoa
cd /scratch/hpc/07/zhang303/tecoa
srun python finetuning.py \
  --batch_size 64 \
  --num_workers 8 \
  --root /scratch/hpc/07/zhang303/tecoa/data \
  --dataset cifar100 \
  --name version_1 \
  --learning_rate 1e-5 \
  --epochs 10 \
  --train_eps 1 \
  --train_numsteps 2 \
  --train_stepsize 1
  #   --imagenet_root /scratch/hpc/07/zhang303/tecoa/data/imagenet1k \