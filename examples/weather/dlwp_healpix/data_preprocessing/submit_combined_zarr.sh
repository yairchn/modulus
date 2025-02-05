#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o ./combined_zarr_processing_%j.out
#SBATCH -e ./combined_zarr_processing_%j.err

SOURCE_FOLDER="${SCRATCH}/"
OUTPUT_FILE="${SCRATCH}/combined_era5_hpx128_1980_2020.zarr"

srun /usr/bin/shifter --env PYTHONUSERBASE=${HOME}/.local/perlmutter/pytorch2.0.1 --entrypoint python3 -u ./combine_zarr_data.py --source_folder $SOURCE_FOLDER --output_file $OUTPUT_FILE
