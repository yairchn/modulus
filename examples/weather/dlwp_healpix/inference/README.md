### INFERENCE:

To run a coupled inference in interactive session:
```
python -W ignore coupled_forecast.py --config=inference_config.json
```
here due to a set of warnings that come up from this code and depndencies, we are using `-W ignore`. In an interactive session, we need:
```
export RANK=0
export WORLD_SIZE=1
export SLURM_NPROCS=1
export HYDRA_FULL_ERROR=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

To run a coupled inference in a slurm job use:
```
sbatch submit_inference.sh
```
Make sure to set the number of GPUs to match that in `inference_config.json`


The inference is using a json config that has a number of options such as 
the checkpoints are given as `atmos_model_checkpoint` and `ocean_model_checkpoint`.
These should be a single checkpoint per model. for checkpoint ensemble see below. 
Mind that shortest lead time for inference of a coupled model is 192h (8 days), it is limited by the coupler requiring 2 steps of the ocean model. 
Mind the `atmos_subset_channels` option in the config. Here `null` will save all channels but if specified as a list, i.e. `["t2m0", "t850", "z500"]` the resulting forecast / hindcast will save to disc only a subset of the channels to save disk space. 

## CHECKPOINT ENSEMBLE

To run an ensemble of checkpoint we use `checkpoint_runner.py` a code that submits many calls to `coupled_forecast.py `:
```
python -W ignore checkpoint_runner.py --config=ckpt_ens_config.json
```
here the config is almost identical but has lists of checkpoints rather than a single checkpoint. `checkpoint_runner.py` will generate all combinations of checkpoints from the various models (ocean and atmosphere), make a config per each ckpt combination and inference all of them. It will split the work on all GPUs available. 
The code can resume a run from where it stopped. When completes it will combines all netcdfs in the parent directory and remove all temp files (see `keep_temp_files` arg in the config). 
To submit `checkpoint_runner.py` as a slurm job (potentially on more than a single node) use  
```
sbatch submit_checkpoint_runner.sh
```

`combine_checkpoints.py` will combine files poer atmos or ocean frm all `ckpt_*` directories to a zarr file in the parent dir. 

## IC PERTURBATIONS ENSEMBLE
Ic ensmbele forecast is only working now via the batch dim from training and is thus limited by mnemory. For HPX64 I see that on Perlmutter I can fit noise_dim=4. To increase the ensmeble further there are two options: 
1. fix the seed and run several times, 
2. disentengle the instantiation of teh coupler and data processing from the model generation and thus separate the data perop step from running the model and run the model for many ICs. 

In the current code, `ensemble_utils.py` we have a bred vector method and a Gaussian method (untested). The bred vector has a wrapper for a centered ensembel that is currently keep the control and half of the perturbed, while overwriting the other half with perturbations that balance the first half. 


### HINDCAST:
An embarrassingly parallel hindcast workflow is suggested here, which can run a hindcast on several nodes. Per node hindcast is simply calling `checkpoint_runner.py` on selected years, leaving the distribution on the available GPUs in the node to `checkpoint_runner.py` to be used for checkpoints. To run hindcast on many nodes:
```
bash serve_hindcast.sh <number_of_jobs>
```
here each node will recieve a fraction of years equal to `<totoal_number_of_years>/<number_of_jobs>`. Many parameters are hard coded in this file. Specifically `START_YEAR` and `END_YEAR` are overriding their counterparts in `hindcast_config.json`. 
To test the hindacast in interactive session:
```
python -W ignore hindcast.py --config=hindcast_config.json
```

once a hindcast is complete:
1. call with `combine_checkpoints.py` for all hindcast years or `run_parallel_combine.py` once. 
2. than call `restructure_hindcast.py` to convert hindcast data to the icmwf format in which each file has a single day of year for all hindcast years. Note `restructure_hindcast.py` might expect already combined `forecast_atmos.py` and `forecast_ocean.py` across checkpoint ensemble if the hindcast is for an ensemble of checkpoints. Modifications might be needed for handling this. 

See README in score folder for computing the bias. 




