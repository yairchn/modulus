#!/usr/bin/env python

import os
import gc
import numpy as np
import xarray as xr

from joblib import Parallel, delayed, parallel_backend
import dask
from dask.diagnostics import ProgressBar
from postprocessing import load_training_data_from_zarr

dask.config.set(scheduler="processes", num_workers=64)
pbar = ProgressBar()
pbar.register()


def load_hindcast_data(hindcast_path):
    """
    Load a hindcast dataset with Dask-based chunking.
    Adjust 'chunks' to your dimension sizes as needed.
    """
    ds = xr.open_dataset(
        hindcast_path, 
        engine="zarr",
        chunks={
            'time': 100,   # Larger chunk in time 
            'lead_time': 1,
            'height': 64, # Try bigger chunk in space if memory allows
            'width': 64
        }
    )
    return ds

def load_verification_data(verification_path, times, variables):
    """
    Example verification data loader with Dask chunking.
    You might need .sel(time=...) or your own logic for the times.
    """
    ds = xr.open_dataset(
        verification_path, 
        engine="zarr",   # or "netcdf4"
        chunks={
            'time': 10,
            'height': 64,
            'width': 64
        }
    )

    # Subset to times of interest (assuming times is a list/array of datetime64)
    ds = ds.sel(time=times, method='nearest')[variables]
    return ds

def get_unique_valid_times(lead_times, times):
    times_numeric = times.astype(np.int64)
    times_reshaped = times_numeric[:, np.newaxis]
    lead_times_reshaped = lead_times.values[np.newaxis, :]
    valid_time_numeric = times_reshaped + lead_times_reshaped
    valid_time_numeric = valid_time_numeric.ravel()
    valid_time_numeric = np.unique(valid_time_numeric)
    valid_time = valid_time_numeric.astype('datetime64[ns]')
    return valid_time

def compute_error_for_year(year, hindcast, verification_data, save_directory):
    """
    Slices hindcast data for `year`, aligns with verification, 
    computes error = hindcast - verification, writes partial netCDF.
    Returns the filepath for this year's error file.
    """
    year_str = str(year)
    year_file = os.path.join(save_directory, f'error_year_{year_str}.nc')
    if os.path.exists(year_file):
        print(f"[compute_error_for_year] Found existing file for {year}, skipping: {year_file}")
        return year_file

    print(f"[compute_error_for_year] Computing error for year {year} ...")
    year_start = f"{year}-01-01"
    year_end   = f"{year}-12-31"

    # Slice hindcast data for this year
    hindcast_year = hindcast.sel(time=slice(year_start, year_end))
    verification_year = verification_data.sel(time=slice(year_start, year_end))
    error = hindcast_year - verification_year

    # Convert to a dataset named "error" if it's a DataArray
    if isinstance(error, xr.DataArray):
        error_ds = error.to_dataset(name="error")
    else:
        error_ds = error
    # Write to netCDF file (triggering Dask computation)
    error_ds.to_netcdf(year_file, engine="netcdf4")

    # Clean up references
    del error_ds
    gc.collect()

    print(f"[compute_error_for_year] Wrote {year_file}")
    return year_file

def compute_bias_for_var(var, error_files, save_directory):
    """
    For a given variable (e.g. 't2m'), load partial 'error_year_*.nc' files,
    compute monthly or daily bias, and write to 'bias_for_<var>.nc'.
    """
    bias_file = os.path.join(save_directory, f"bias_for_{var}.nc")
    if os.path.exists(bias_file):
        print(f"[compute_bias_for_var] Skipping {var}, file exists: {bias_file}")
        return bias_file

    print(f"[compute_bias_for_var] Computing bias for var {var} ...")

    datasets = []
    for ef in error_files:
        if not os.path.exists(ef):
            continue
        ds = xr.open_dataset(ef, chunks={'time': 10, 'height': 64, 'width': 64})
        if var not in ds.data_vars:
            ds.close()
            continue
        datasets.append(ds[[var]])

    if not datasets:
        print(f"[compute_bias_for_var] No data found for {var}, skipping.")
        return None

    # Concatenate along 'time' dimension
    combined_error = xr.concat(datasets, dim="time")
    # Free memory for partial
    for ds in datasets:
        ds.close()

    dayofyear = combined_error['time'].dt.dayofyear
    bias_ds = combined_error.groupby(dayofyear).mean(dim="time", skipna=True)

    bias_ds.to_netcdf(bias_file, engine="netcdf4")

    combined_error.close()
    bias_ds.close()
    gc.collect()

    print(f"[compute_bias_for_var] Wrote {bias_file}")
    return bias_file

def load_hindcast_from_zarr(zarr_path):
    ds = xr.open_zarr(zarr_path).rename({'step': 'lead_time'}).isel(ckpt_ensemble=0).isel(ic_ensemble=0)
    ds_selected = xr.Dataset(
        {
            var: ds[var].transpose('time', 'lead_time', 'face', 'height', 'width')
            for var in ds.data_vars
        },
        coords={
            'time': ds['time'].values,
            'lead_time': ds['lead_time'].values,
            'face': ds['face'].values,
            'height': ds['height'].values,
            'width': ds['width'].values,
        }
    )
    return ds_selected

def main():
    
    n_jobs = 64

    # Paths
    zarr_path = '/global/cfs/projectdirs/m4331/datasets/era5_hpx/era5_hpx64_1980.zarr'
    hindcast_path = "/global/cfs/projectdirs/nvdlesm/inference/v0/atmos_ocean/default/hpx64/8channels/uw_config/lr_5e_4/atmos_0262_ocean_0431/hindcast/biweekly/60day/combined_forecast_atmos_new.zarr"
    save_directory = os.path.dirname(hindcast_path)+"/tmp3/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    print("Loading hindcast data...")
    hindcast = load_hindcast_from_zarr(hindcast_path)
    variables_to_remove = ['tau300-700', 'tcwv', 'ws10m', 'z1000', 'z250']
    variables = [var for var in hindcast.data_vars.keys() if var not in variables_to_remove]
    hindcast = hindcast[variables]

    print("[main] Preparing verification data...")
    lead_times = (hindcast.lead_time * 1e9).astype('timedelta64[ns]')
    hindcast = hindcast.assign_coords(lead_time=lead_times)
    times = hindcast.time.values
    unique_times = get_unique_valid_times(lead_times, times)

    # Suppose you have your own function that loads verification for these times & vars:
    verification_data = load_training_data_from_zarr(zarr_path, unique_times, variables)
    verification_data = verification_data[variables]

    # 4.2: Compute error for each year in parallel
    years = np.unique(hindcast["time"].dt.year.values)
    def run_error(year):
        return compute_error_for_year(year, hindcast, verification_data, save_directory)

    print("[main] Computing error by year...")
    with parallel_backend("loky", n_jobs=n_jobs):
        error_files = Parallel()(
            delayed(run_error)(y) for y in years
        )
    error_files = [ef for ef in error_files if ef is not None]

    # 6.2: Compute bias in parallel, one variable at a time
    def run_bias(var):
        return compute_bias_for_var(var, error_files, save_directory)

    print("[main] Computing bias by variable...")
    with parallel_backend("loky", n_jobs=n_jobs):
        bias_files = Parallel()(
            delayed(run_bias)(var) for var in variables
        )
    bias_files = [bf for bf in bias_files if bf is not None]

    # 6.3: Merge all bias files into a single dataset
    merged_path = os.path.join(save_directory, "final_bias.nc")
    if not os.path.exists(merged_path):
        ds_list = []
        for bf in bias_files:
            if bf is None:
                continue
            ds_bf = xr.open_dataset(bf)
            ds_list.append(ds_bf)

        if ds_list:
            print(f"[main] Merging {len(ds_list)} bias files into one final dataset...")
            final_ds = xr.merge(ds_list)
            final_ds.to_netcdf(merged_path)
            final_ds.close()
            # Close partial datasets
            for ds_i in ds_list:
                ds_i.close()
            print(f"[main] Merged bias to: {merged_path}")
        else:
            print("[main] No bias files to merge.")
    else:
        print(f"[main] Final merged file already exists: {merged_path}")

    print("[main] All done.")


if __name__ == "__main__":
    main()
