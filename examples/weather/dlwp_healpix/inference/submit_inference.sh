#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o hpx64_8var_coupled_inference_%j.out
#SBATCH -e hpx64_8var_coupled_inference_%j.err
#SBATCH -q preempt

export HYDRA_FULL_ERROR=1

# make sure this path is correct for your NV-dlesm git folder
cd ${HOME}/NV-dlesm/inference/
srun --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 shifter --entrypoint python3 -W ignore coupled_forecast.py  --config=inference_config_hpx64.json