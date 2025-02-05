#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH -o ./run_hindcast_%j.out
#SBATCH -e ./run_hindcast_%j.err
#SBATCH -q preempt

set -x

CONFIG_PATH="${HOME}/NV-dlesm/inference/hindcast_config.json"
SCRIPT_DIR="${HOME}/NV-dlesm/inference"

cd ${SCRIPT_DIR}

# Check if year arguments are provided
if [ $# -eq 2 ]; then
    # If both start and end years are provided, use them
    srun --nodes 1 --ntasks-per-node 1 --gpus-per-node 4 shifter --entrypoint python3 hindcast.py \
        --config=${CONFIG_PATH} \
        --start-year=$1 \
        --end-year=$2
else
    # If no years provided, run with defaults from config
    srun --nodes 1 --ntasks-per-node 1 --gpus-per-node 4 shifter --entrypoint python3 hindcast.py \
        --config=${CONFIG_PATH}
fi