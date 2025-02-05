#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus=8
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o ens_inference_%j.out
#SBATCH -e ens_inference_%j.err
export HYDRA_FULL_ERROR=1

# make sure this path is correct for your NV-dlesm git folder
cd ${HOME}/NV-dlesm/inference/
srun --nodes 2 --ntasks-per-node 4 --gpus-per-node 4 shifter --entrypoint python -W ignore checkpoint_runner.py --config=inference_config_hpx64_ens_test.json
