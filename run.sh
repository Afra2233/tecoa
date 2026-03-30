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
srun python finetuning.py --batch_size 64 --root /scratch/hpc/07/zhang303/tecoa/data --dataset ImageNet --name feimogu --train_eps 1 --train_numsteps 2 --train_stepsize 1