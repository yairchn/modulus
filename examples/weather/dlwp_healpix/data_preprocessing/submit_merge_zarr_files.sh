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
#SBATCH -o merge_zarr_file_processing_%j.out
#SBATCH -e merge_zarr_file_processing_%j.err

SOURCE_FOLDER=$1
TARGET_FOLDER=$2

if [ -z "$SOURCE_FOLDER" ]; then
    echo "Error: Source folder not specified."
    echo "Usage: sbatch $0 /path/to/yearly/zarr/files /path/to/target/"
    exit 1
fi


if [ -z "$TARGET_FOLDER" ]; then
    echo "Error: Source folder not specified."
    echo "Usage: sbatch $0 /path/to/yearly/zarr/files /path/to/target/"
    exit 1
fi

srun /usr/bin/shifter --env PYTHONUSERBASE=${HOME}/.local/perlmutter/pytorch2.0.1 --entrypoint python3 ./merge_zarr_files.py --source_folder $SOURCE_FOLDER --target_store $TARGET_FOLDER
