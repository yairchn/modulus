#!/bin/bash

#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o ./compute_bias%j.out
#SBATCH -e ./compute_bias%j.err
#SBATCH -q preempt

set -x

srun /usr/bin/shifter --entrypoint python3 save_model_bias.py
