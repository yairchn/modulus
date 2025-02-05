#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus=32
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o hpx64_8var_coupled_training_%j.out
#SBATCH -e hpx64_8var_coupled_training_%j.err
#SBATCH --dependency=afterany:31655215
#SBATCH -q preempt

export HYDRA_FULL_ERROR=1

# make sure this path is correct for your NV-dlesm git folder
cd ${HOME}/NV-dlesm/
srun --nodes 8 --ntasks-per-node 4 --gpus-per-node 4 shifter --entrypoint python3 train.py --config-name config_hpx64_coupled_dlwp
