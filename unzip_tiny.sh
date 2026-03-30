#!/bin/bash
#SBATCH --job-name=tinyimage_download
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8


srun /storage/hpc/07/zhang303/conda_envs/tecoa/bin/python download_tinyimagenet.py