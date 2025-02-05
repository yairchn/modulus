#!/usr/bin/env python

import os
import gc
import glob
import numpy as np
import xarray as xr
from dask import config as dask_config


def load_training_data_from_zarr(zarr_path, variable):
    ds = xr.open_zarr(zarr_path)

    data_vars = {}
    variable_data = ds['inputs'].sel(channel_in=variable)
    data_vars[variable] = variable_data.transpose('time', 'face', 'height', 'width')
    
    new_height_coords = np.arange(data_vars[variable]['height'].size)
    new_width_coords = np.arange(data_vars[variable]['width'].size)

    ds_selected = xr.Dataset(
        data_vars,
        coords={
            'time': variable_data['time'],
            'face': variable_data['face'],
            'height': new_height_coords,
            'width': new_width_coords,
            'lat': (['face', 'height', 'width'], ds['lat'].values),
            'lon': (['face', 'height', 'width'], ds['lon'].values),
        }
    )
    return ds_selected


def main():
    dask_config.set(scheduler="processes", num_workers=8)
    
    variable = "z500"
    hindcast_dir = f"/global/cfs/projectdirs/m4935/nvdlesm/ensemble_atmos_98_good_ocean_Jan15_healpix_dir/reformatted/{variable}/"
    verification_zarr_path = "/global/cfs/projectdirs/m4331/datasets/era5_hpx/era5_hpx64_1980.zarr"
    output_dir = "/global/cfs/projectdirs/m4935/nvdlesm/ensemble_atmos_98_good_ocean_Jan15_healpix_dir/bias/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pattern = os.path.join(hindcast_dir, "hindcast_*.nc")
    hindcast_files = sorted(glob.glob(pattern))
    print(f"Found {len(hindcast_files)} day-of-year hindcast files.")


    daily_bias_files = []
    for hfile in hindcast_files:
        base_name = os.path.basename(hfile).replace("hindcast_", "").replace(".nc", "")
        out_name = f"bias_{base_name}_{variable}.nc"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            print(f"Skipping {hfile} because {out_path} already exists.")
            daily_bias_files.append(out_path)  # Keep track of it anyway for merging
            continue

        print(f"\n--- Processing {hfile} ---")
        
        hindcast_ds = xr.open_dataset(hfile, chunks={"face": 1})
        
        verification_times = hindcast_ds.time.expand_dims(step=hindcast_ds.step) + hindcast_ds.step
        with xr.open_zarr(verification_zarr_path) as era5:
            verif_aligned_da = era5.sel(
                channel_in=variable, 
                channel_out=variable, 
                channel_c='land_sea_mask', 
                time=verification_times
            )
            verif_aligned_da['time'] = hindcast_ds.time


        bias_da = (hindcast_ds - verif_aligned_da['inputs']).mean(dim="time").compute()

        print(f"Writing daily bias to {out_path} ...")
        bias_da.to_netcdf(out_path, engine="netcdf4")

        hindcast_ds.close()
        bias_da.close()
        gc.collect()

        daily_bias_files.append(out_path)

    print("\n--- Done computing bias for the selected day-of-year files ---")

    merged_file = os.path.join(output_dir, f"bias_all_days_{variable}.nc")
    print("Merging all daily bias files...")
    ds_list = []
    for bf in daily_bias_files:
        ds_temp = xr.open_dataset(bf)
        bname = os.path.basename(bf)
        mmdd = bname.replace("bias_", "").replace(".nc", "")
        day_of_year = int(mmdd[:2]) * 31 + int(mmdd[3:5])
        ds_temp = ds_temp.assign_coords({"day_of_year": day_of_year})
        ds_list.append(ds_temp)

        combined_bias = xr.concat(ds_list, dim="day_of_year")
        combined_bias.to_netcdf(merged_file)

        for ds_tmp in ds_list:
            ds_tmp.close()
        combined_bias.close()
        print(f"Merged all daily biases into {merged_file}")
    else:
        print(f"Already found merged file {merged_file} - skipping merge.")

    print("All finished!")

if __name__ == "__main__":
    main()
