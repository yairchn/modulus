#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o ./stats_processing_%j.out
#SBATCH -e ./stats_processing_%j.err

set -x
SOURCE_FILE=$1
OUTPUT_FILE=$2

if [ -z "$SOURCE_FILE" ]; then
    echo "Error: Target folder not specified."
    echo "Usage: sbatch $0 /path/zarr/file </path/to/output/file>"
    exit 1
fi

if [ -z "$OUTPUT_FILE" ]; then
     srun /usr/bin/shifter --env PYTHONUSERBASE=${HOME}/.local/perlmutter/pytorch2.0.1 --entrypoint python3 ./calculate_stats.py --source $SOURCE_FILE
else
     srun /usr/bin/shifter --env PYTHONUSERBASE=${HOME}/.local/perlmutter/pytorch2.0.1 --entrypoint python3 ./calculate_stats.py --source $SOURCE_FILE --output_file $OUTPUT_FILE
fi
