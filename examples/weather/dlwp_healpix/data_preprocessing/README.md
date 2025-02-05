# Data Preprocessing for Earth2Studio

This folder contains the files needed for generating new training data using Earth2Grid and Earth2Studio for re-gridding and data download/processing.

## Prerequisites

Ensure that Earth2Grid and Earth2Studio are installed. 
For earth2grid, follow the README in the repo:
https://github.com/NVlabs/earth2grid
For Earth2Studio, we are using a specific branch on (pending David's fork branch)

you need to clone the repo, add remote and checkout the branch 

https://github.com/daviddpruitt/earth2studio/tree/era5_data_processing

and than install in the `earth2studio` folder with  `pip install .`

## Some common troubleshooting steps:
 
`pip install --upgrade fsspec`
`pip install apache_beam`
`pip install healpy`
`pip install metpy`

`git clone https://github.com/NVlabs/earth2grid.git`
`cd earth2grid`
`pip install .`

if after installing packages you see have import issues in a slurm job, but notin an interactive session try removing the assinment of PYTHONUSERBASE below from the srun call in the slurm scripts
`--env PYTHONUSERBASE=${HOME}/.local/perlmutter/pytorch2.0.1`

## Workflow

The code generates individual years of data across many jobs, computes derived channels on these yearly Zarr files, and finally combines the Zarr files into a single Zarr file for all years.

### Steps

1. Generate yearly data:
   ```bash
   bash prepare_data.sh </path/to/yearly/zarr/files>  <level>
   ```
   This script submits a Slurm job per year using `submit_nvdlesm_data.sh`. The range of years (1980 to 2020) is coded in `prepare_data.sh` and can be changed there. Here `</path/to/yearly/zarr/files>` could be `$SCRATCH`. The `level` argument correspnding to the healpix grid such that `2**level` is the healpix resolution; so that level = 5,6,7 corresponds to HPX32, HPX64 and HPX128 respectivey.

2. Add derived channels:
   ```
   sbatch submit_derived_channels.sh </path/to/yearly/zarr/files>
   ```
   This step produces derived channels from existing ones, such as tau (geopotential thickness between levels) and windspeed.

3. Add trailing mean channels:
   ```
   sbatch submit_trailing_mean.sh </path/to/yearly/zarr/files>
   ```
   This step produces trailing mean of existing channels for ocean coupled model, such as ws10m and z1000. For a coupled model, the ocean model needs the atmosphere channels averaged on 48h window.

4. Combine all years into a single Zarr dataset:
   ```
   sbatch submit_combined_zarr.sh
   ```

## File Descriptions

- `prepare_data.sh __/path/to/yearly/zarr/files__ `: Main script to initiate the data generation process.
- `submit_nvdlesm_data.sh __--year <year_to_preprocess>__ __--path </path/to/yearly/zarr/files>__`: Slurm job script for submitting individual year processing jobs.
- `build_dataset.py __--year <year_to_preprocess>__ __--path </path/to/yearly/zarr/files>__ [--level <healpix resolution level>] `: Python script that downloads and processes data for a specific year.
- `add_derived_channels.py __--target_folder </path/to/yearly/zarr/files>__`: Python script to add derived channels to the yearly Zarr files.
- `compute_trailing_mean.py __--target_folder </path/to/yearly/zarr/files>__ [--variables_name <channals name to compute trailing mean>] __ [--coupled_dt <time resoluton of the atmos model to be coupled>] __ [--influence_window <range for trailing mean>]`: Python script to add trailing mean of existing channels to the yearly Zarr files.
- `combine_zarr_data.py`: Python script to combine all yearly Zarr files into a single dataset.
- `submit_derived_channels.sh`: Slurm job script for adding derived channels.
- `submit_trailing_mean.sh`: Slurm job script for adding trailing mean of existing channels.
- `submit_combined_zarr.sh`: Slurm job script for combining all Zarr files.

## Checkpointing

`build_dataset.py` writes to a checkpointing log for completed years. `prepare_data.sh` handles checkpointing to see if some files did not complete and repeats them if necessary.

## Adding additional fields

Additional fields can be added to existing datasets. Steps to add channels to an existing dataset:

1. Generate yearly data [follow the steps above](#Steps). The dates in the target dataset need to exactly match the dates available in the dates being generated, as does the resolution.
   Merge step #4 should be skipped as this will done automatically in the next step.

3. Merge the datasets together:
   ```bash
   bash submit_merge_zarr_files.sh </path/to/yearly/zarr/files> </path/to/target/store>
   ```
   This script submits a Slurm jub that takes the individual yearly files and appends the channel data to the target store channel data. 

## TODO: 
- The resulting zarr file should have in the name the years, resolution and number of channels 
- Consider adding a climatology calcualtion code so that each data has a climatology next to it. 
