#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o hpx_data_processing_%j.out
#SBATCH -e hpx_data_processing_%j.err

# make sure this path is correct for your NV-dlesm git folder
srun shifter --env PYTHONUSERBASE=${HOME}/.local/perlmutter/pytorch2.0.1 --entrypoint python3 build_dataset.py --year ${1} --path ${2} --level ${3}
