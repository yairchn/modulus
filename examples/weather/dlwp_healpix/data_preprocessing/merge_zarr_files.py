import zarr
import numpy as np
import pandas as pd
import argparse
from glob import glob
import os

def merge_channels(source_folder, target_file, chunk_size):
    # Find all yearly Zarr files
    target_ds = zarr.open(target_file)
    zarr_files = sorted(glob(os.path.join(source_folder, "era5_hpx*_????.zarr")))
    
    if not zarr_files:
        raise ValueError(f"No Zarr files found in {source_folder}")

    print(f'Processing {len(zarr_files)} files')

    # verify that *all* the source files have the appriate channels and all the timesteps
    spatial_resolution = target_ds.inputs.shape[-3:]
    input_times = []

    for zarr_file in zarr_files:
        store = zarr.open(zarr_file)
        if store.inputs.shape[-3:] != spatial_resolution:
            print(f"Spatial resolutions don't match for: {zarr_file}")
            return
        input_times.extend(store.time[736:])

    # Convert the list of time dimensions to a numpy array
    input_times = np.array(input_times)
    target_times = np.array(target_ds.time)

    # Find any differences between the time dimensions
    differences = np.setdiff1d(input_times, target_times)

    # Print the differences
    if len(differences) > 0:
        print(f"The following time dimensions are not present in the target dataarray: {differences}")
        print(f"Exiting to prevent data corruption")
        return

    first_ds = zarr.open(zarr_files[0], mode='r')
    num_old_channels = len(target_ds.channel_in)
    target_ds.channel_in.append(first_ds.channel_in[:])
    target_ds.channel_out.append(first_ds.channel_out[:])

    old_shape = target_ds.inputs.shape
    new_shape = (len(target_times), len(target_ds.channel_in)) +  spatial_resolution

    # reshape the target dataarrays to
    target_ds.inputs.resize(new_shape)
    target_ds.targets.resize(new_shape)

    # Combine time-dependent data
    time_index = 0
    time_offset = 0 # used if there's extra dates in the files

    for file in zarr_files:
        ds = zarr.open(file, mode='r')
        time_steps = ds.inputs.shape[0]
        print(f"Processing {file}")
        # using chunking to improve throughput and avoid OOM errors
        for chunk_start in range(time_offset, time_steps, chunk_size):
            chunk_end = min(chunk_start+chunk_size, time_steps)
            root_start = time_index + chunk_start - time_offset
            root_end = time_index + chunk_end - time_offset
            target_ds.inputs[root_start:root_end,num_old_channels:] = ds.inputs[chunk_start:chunk_end,:]
            target_ds.targets[root_start:root_end,num_old_channels:] = ds.targets[chunk_start:chunk_end,:]
        
        time_index += time_steps - time_offset

    # Consolidate metadata
    zarr.consolidate_metadata(target_file)
    print(f"Variables add to zarr file {target_file}")


def main():
    parser = argparse.ArgumentParser(description="Add all the variables from a list of yearly zarr stores into an existing target zarr..")
    parser.add_argument("--source_folder", type=str, required=True, help="Path to the folder containing yearly Zarr stores with variables to add")
    parser.add_argument("--target_store", type=str, required=True, help="Path to the target Zarr store")
    parser.add_argument("--chunk_size", type=int, default=384, help="Chunk size to use for copying data")
    args = parser.parse_args()

    merge_channels(args.source_folder, args.target_store, args.chunk_size)

if __name__ == "__main__":
    main()
