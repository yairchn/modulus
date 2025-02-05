#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o trailing_mean_processing_%j.out
#SBATCH -e trailing_mean_processing_%j.err

TARGET_FOLDER=$1

if [ -z "$TARGET_FOLDER" ]; then
    echo "Error: Target folder not specified."
    echo "Usage: sbatch $0 /path/to/yearly/zarr/files"
    exit 1
fi

srun /usr/bin/shifter --env PYTHONUSERBASE=${HOME}/.local/perlmutter/pytorch2.0.1 --entrypoint python3 ./compute_trailing_mean.py --target_folder $TARGET_FOLDER --variables_name ws10m z1000
