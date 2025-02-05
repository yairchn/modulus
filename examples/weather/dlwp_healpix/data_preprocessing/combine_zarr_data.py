import zarr
import numpy as np
import pandas as pd
import argparse
from glob import glob
import os

def combine_zarr_files(source_folder, output_file, slice_size):
    # Find all yearly Zarr files
    zarr_files = sorted(glob(os.path.join(source_folder, "era5_hpx*_????.zarr")))
    
    if not zarr_files:
        raise ValueError(f"No Zarr files found in {source_folder}")

    print(f'Processing {len(zarr_files)} files')

    # Open the first file to get metadata
    first_ds = zarr.open(zarr_files[0], mode='r')
    
    # Create the output Zarr store
    store = zarr.DirectoryStore(output_file)
    root = zarr.group(store=store)

    # Initialize datasets in the output file
    total_time_steps = sum(zarr.open(f, mode='r').inputs.shape[0] for f in zarr_files)
    
    # redo chunk size, adding derived channels can sometimes cause strange issues 
    # this forces each channel to be its own chunks and puts 4 time steps in a single chunk to reduce
    # the overall number of chunks
    chunk_size = (4, 1, *first_ds.inputs.chunks[2:])

    root.create_dataset('inputs', shape=(total_time_steps, *first_ds.inputs.shape[1:]), fill_value=float("NaN"), 
                        chunks=chunk_size, dtype=first_ds.inputs.dtype, overwrite=True)
    root.create_dataset('targets', shape=(total_time_steps, *first_ds.targets.shape[1:]), fill_value=float("NaN"), 
                        chunks=chunk_size, dtype=first_ds.targets.dtype, overwrite=True)

    root.inputs.attrs["_ARRAY_DIMENSIONS"] = ["time", "channel_in", "face", "height", "width"]
    root.inputs.attrs["coordinates"] = "level lon lat"
    root.targets.attrs["_ARRAY_DIMENSIONS"] = ["time", "channel_out", "face", "height", "width"]
    root.targets.attrs["coordinates"] = "level lon lat"

    # Copy over constant data and metadata
    for key in ['constants', 'channel_in', 'channel_out', 'channel_c', 'face', 'width', 'height', 'lat', 'lon']:
        root.create_dataset(key, data=first_ds[key][:], chunks=first_ds[key].chunks, dtype=first_ds[key].dtype,
                            fill_value=first_ds[key].fill_value, overwrite=True)
        root[key].attrs.update(first_ds[key].attrs)

    # set fill value so xarray interprets correctly
    root['constants'].fill_value=float("NaN")

    # Combine time-dependent data
    time_index = 0
    all_times = []

    for file in zarr_files:
        ds = zarr.open(file, mode='r')
        time_steps = ds.inputs.shape[0]
        
        print(f"Processing {file}")
        # using slicing to improve throughput and avoid OOM errors
        for slice_start in range(0, time_steps, slice_size):
            slice_end = min(slice_start+slice_size, time_steps)
            root_start = time_index + slice_start
            root_end = time_index + slice_end
            root.inputs[root_start:root_end] = ds.inputs[slice_start:slice_end]
            root.targets[root_start:root_end] = ds.targets[slice_start:slice_end]
        
        all_times.extend(ds.time[:])
        time_index += time_steps

    # Create combined time dataset
    root.create_dataset('time', data=np.array(all_times), chunks=first_ds.time.chunks, overwrite=True)
    root.time.attrs.update(first_ds.time.attrs)

    # Update global attributes
    root.attrs['start_date'] = pd.Timestamp(all_times[0]).strftime('%Y-%m-%d')
    root.attrs['end_date'] = pd.Timestamp(all_times[-1]).strftime('%Y-%m-%d')
    root.attrs['total_time_steps'] = total_time_steps

    # Consolidate metadata
    zarr.consolidate_metadata(store)
    print(f"Combined Zarr file created at {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine yearly Zarr files into a single dataset.")
    parser.add_argument("--source_folder", type=str, required=True, help="Path to the folder containing yearly Zarr files")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output combined Zarr file")
    parser.add_argument("--slice_size", type=int, default=96, help="Number of time steps to copy at once")
    args = parser.parse_args()

    combine_zarr_files(args.source_folder, args.output_file, args.slice_size)

if __name__ == "__main__":
    main()
