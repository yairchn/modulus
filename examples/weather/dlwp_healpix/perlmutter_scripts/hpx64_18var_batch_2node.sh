#!/bin/bash
#SBATCH --account=<account to run under>
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=nersc/pytorch:ngc-23.07-v1
#SBATCH -o hpx64_18var_%j.out 
#SBATCh -e hpx64_18var_%j.err
#SBATCH -q preempt

set -x

readonly python_base="/global/homes/d/${USER}/.local/perlmutter/dlwp"
readonly config="config_symmetric_conv_next_18ch"
readonly expmt_name="hpx64_18var_test"
readonly expmt_dir=<where you are going to store the experiment (checkpoints and logging info)>
readonly batch_size=8

# all the overriden options
readonly train_opts="experiment_name=${expmt_name} output_dir=${expmt_dir} batch_size=4"

cd ${HOME}/NV-dlesm/
srun --nodes 2 --ntasks-per-node 4 --gpus-per-node 4 shifter --env PYTHONUSERBASE=${python_base}  --entrypoint python3 -u train.py --config-name=${config} ${train_opts}
