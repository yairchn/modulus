# Deep Learning Weather Prediction (DLWP) model for weather forecasting

This example is an implementation of the coupled Ocean-Atmosphere DLWP model.

## Problem overview

The goal is to train an AI model that can emulate the state of the atmosphere and predict
global weather over a certain time span. The Deep Learning Weather Prediction (DLWP) model
uses deep CNNs for globally gridded weather prediction. DLWP CNNs directly map u(t) to
its future state u(t+Δt) by learning from historical observations of the weather,
with Δt set to 6 hr. The Deep Learning Ocean Model (DLOM) that is designed to couple with
deep learning weather prediction (DLWP) model. The DLOM forecasts sea surface
temperature (SST). DLOMs use deep learning techniques as in DLWP models but are
configured with different architectures and slower time stepping. DLOMs and DLWP models
are trained to learn atmosphere-ocean coupling.

## Getting Started on EOS

To train the coupled DLWP model, run

```bash
python train.py --config-name config_hpx32_coupled_dlwp
```

To train the coupled DLOM model, run

```bash
python train.py --config-name config_hpx32_coupled_dlom
```

## For perlmutter
## Description of files

`trainer.py`: Includes the definition of the trainer object used to create DLWP-style models.<br>
`train.py`: Definition of the training routine used for compiling and training DLWP-style models.<br>
`utils.py`: Code for miscellaneous functions used in DLWP.<br>
`rmse_acc.py`: Routine for calculating the root mean squared error and anomaly correlation coefficient for DLWP simulations.

see `perlmutter_scripts` for examples. 
For the coupled model, there is a PR in modulus for allowing the improt of the 
coupled datapipes (https://github.com/NVIDIA/modulus/pull/681). 
Until it is merged, training the coupled model requires:
1. cloning modulus
2. adding Yair's fork as a remote
3. checkoing out the branch
4. installing the local code with `pip install .`


## Getting started on Perlmutter

### Prerequsites
- Get added to the project account on Perlmutter. 

**Step 1**: Clone the repository `git clone https://github.com/AtmosSci-DLESM/NV-dlesm.git`

### Running Interactive Job 

**Step 2**: Start an interactive session on 1 node and 4 gpus

``salloc --account=m4331  --module=gpu,nccl-2.18 --nodes 1 --ntasks-per-node 1 --constraint gpu --gpus-per-node 4 --qos interactive --time 01:00:00``

**Step 3**: Run shifter image ``shifter --image=nersc/pytorch:ngc-23.07-v1`` and check pip installation with ``which pip``. Ensure its ``/usr/local/bin/pip``

**Step 4**: Set environment variables 

```
export RANK=0
export WORLD_SIZE=1
export SLURM_NPROCS=1
export HYDRA_FULL_ERROR=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

**Step 5**: Install missing libraries 

```
pip install hydra-core --upgrade
pip install termcolor mlflow
pip install pyproj
pip install nvidia-modulus
pip install protobuf==3.20.0
```

**Step 6**: Set paths and point to config files 

```
readonly python_base="/global/homes/u/user/.local/perlmutter/dlwp"
readonly config="config_symmetric_conv_next_18ch" # Train 18 channel model
readonly expmt_name="hpx64_18var_test"  # Experiment name
readonly expmt_dir=/pscratch/sd/j/jvam/nvesm/ # Path to store model checkpoints and logs
readonly batch_size=8 # Batch size for model training
readonly train_opts="experiment_name=${expmt_name} output_dir=${expmt_dir} batch_size=${batch_size}"
```

**Step 7**: Navigate to the directory containing training script and run `train.py`

``cd /global/homes/u/user/NV-dlesm/weather/dlwp_healpix/``

``torchrun train.py --config-name=${config} ${train_opts}``

### Running batch job 

Batch jobs are run by submitting a job script to the scheduler with the `sbatch` command. The job script contains the commands needed to set up your environment and run your application. Add the code block below to a file `training.sh` and run `sbatch training.sh` More sample scripts can found at `./perlmutter_scripts`

```
#!/bin/bash
#SBATCH --account=m4331
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --module=gpu,nccl-2.18
#SBATCH --image=nersc/pytorch:ngc-23.07-v1
#SBATCH -o hpx64_18var_%j.out 
#SBATCh -e hpx64_18var_%j.err

# Config files and paths
set -x
readonly python_base="/global/homes/u/user/.local/perlmutter/dlwp"
readonly config="config_symmetric_conv_next_18ch"
readonly expmt_name="hpx64_18var_test"
readonly expmt_dir="/pscratch/sd/u/user/nvesm/"

# set environment variables
export RANK=1
export WORLD_SIZE=1
export SLURM_NPROCS=1
export HYDRA_FULL_ERROR=1

# install needed components
srun -n 1 shifter pip install hydra-core --upgrade
srun -n 1 shifter pip install termcolor mlflow
srun -n 1 shifter pip install pyproj
srun -n 1 shifter pip install nvidia-modulus
srun -n 1 shifter pip install protobuf==3.20.0
 

# run training
cd /global/homes/u/user/NV-dlesm/weather/dlwp_healpix/
srun -n 4 shifter python3 -u train.py --config-name=${config} experiment_name=${expmt_name} output_dir=${expmt_dir} batch_size=4
```

### Visualizing loss curve 
Due to port issues on perlmutter, we recamand using tensorboard from your local machine. First download contents to the experiment directory `/pscratch/sd/u/user/nvesm/` to your local machine, and then run tensorboard to visualize the logs with: 
`tensorboard --logdir ./tensorboard/` 
To install tensorboard run `pip install tensorboard`

# Inference
see inference/README.md

# Data Preprocessing
see data_preprocessing/README.md