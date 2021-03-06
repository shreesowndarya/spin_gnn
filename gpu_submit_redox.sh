#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=redox_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/gpu.%j.out

source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu

srun python train_model_redox.py
