import os
import glob
import xarray as xr
import numpy as np 
import dask
from dask.diagnostics import ProgressBar
import argparse
import json
import gc
from tqdm import tqdm

def find_completed_checkpoints(base_dir):
    """
    Find all checkpoint directories that have completed.txt and the expected output files.
    """
    completed_dirs = []
    for ckpt_dir in sorted(glob.glob(os.path.join(base_dir, "ckpt_*"))):
        if (os.path.exists(os.path.join(ckpt_dir, "completed.txt")) and
            os.path.exists(os.path.join(ckpt_dir, "forecast_atmos.nc")) and
            os.path.exists(os.path.join(ckpt_dir, "forecast_ocean.nc"))):
            completed_dirs.append(ckpt_dir)
    return completed_dirs

def process_single_type(completed_dirs, file_type, output_path, checkpoint_dir=None):
    """
    Process forecast files for one type (e.g. 'atmos' or 'ocean') in chunks.
    Each chunk is concatenated and immediately written out as a Zarr checkpoint.
    At the end, all checkpoint chunks are re-opened and concatenated into the final output.
    """
    # Create a checkpoint directory (organized by file type) if not provided.
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(output_path), "checkpoints", file_type)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Open a sample file to extract template attributes.
    first_file = os.path.join(completed_dirs[0], f"forecast_{file_type}.nc")
    with xr.open_dataset(first_file, chunks={}) as first_ds:
        template_attrs = first_ds.attrs.copy()

    # Process files in chunks.
    chunk_size = 8  # Adjust based on your memory constraints or performance needs.
    num_chunks = (len(completed_dirs) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        ckpt_path = os.path.join(checkpoint_dir, f"{file_type}_chunk_{chunk_idx}.zarr")
        if os.path.exists(ckpt_path):
            print(f"Checkpoint for chunk {chunk_idx} exists at {ckpt_path}. Skipping this chunk.")
            continue

        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(completed_dirs))
        chunk_dirs = completed_dirs[start_idx:end_idx]
        chunk_datasets = []
        print(f"Processing chunk {chunk_idx}: indices {start_idx} to {end_idx - 1}")
        for i, ckpt_dir in enumerate(chunk_dirs):
            ds_path = os.path.join(ckpt_dir, f"forecast_{file_type}.nc")
            ds = xr.open_dataset(ds_path, chunks={'time': 8})
            ds = ds.expand_dims("ckpt_ensemble")
            ds = ds.assign_coords(ckpt_ensemble=np.array([start_idx + i]))
            
            # Diagnostic print (already in your code)
            ensemble_vals = ds.coords['ensemble'].values
            print(f"File: {ds_path}")
            print("Ensemble coordinate values:", ensemble_vals)
            print("Unique ensemble values:", np.unique(ensemble_vals))
            
            # If the ensemble coordinate is not [0, 1, 2], override it.
            if not np.array_equal(np.unique(ensemble_vals), np.array([0, 1, 2])):
                print(f"Overwriting ensemble coordinate for file: {ds_path}")
                ds = ds.assign_coords(ensemble=np.array([0, 1, 2]))
            
            # (Optional) Reassign the "face" coordinate.
            ds = ds.assign_coords(face=("face", np.arange(ds.sizes['face'])))
            chunk_datasets.append(ds)
            
        if chunk_datasets:
            # Concatenate the datasets along the new "ckpt_ensemble" dimension.
            chunk_combined = xr.concat(chunk_datasets, dim='ckpt_ensemble')
            # Explicitly set the "ckpt_ensemble" coordinate so it gets saved.
            chunk_combined = chunk_combined.set_coords("ckpt_ensemble")
            print(f"Writing checkpoint for chunk {chunk_idx} to {ckpt_path}")
            with ProgressBar():
                chunk_combined.to_zarr(ckpt_path, mode='w')
            for ds in chunk_datasets:
                ds.close()
            chunk_combined.close()

    # Verify that all checkpoint files exist.
    ckpt_paths = [os.path.join(checkpoint_dir, f"{file_type}_chunk_{i}.zarr")
                  for i in range(num_chunks)]
    missing_ckpts = [p for p in ckpt_paths if not os.path.exists(p)]
    if missing_ckpts:
        raise ValueError(f"Missing checkpoint files: {missing_ckpts}. Job did not complete all chunks.")

    # Re-open all checkpoint chunks and concatenate them.
    print(f"Combining {num_chunks} checkpoint chunks for {file_type} into the final dataset.")
    ds_list = [xr.open_zarr(p) for p in ckpt_paths]
    # (Optional) Ensure each dataset carries the coordinate 'ckpt_ensemble'
    for i, ds in enumerate(ds_list):
        if 'ckpt_ensemble' not in ds.coords:
            ds = ds.assign_coords(ckpt_ensemble=np.arange(ds.dims['ckpt_ensemble']))
            ds_list[i] = ds

    combined_ds = xr.concat(ds_list, dim='ckpt_ensemble')
    combined_ds.attrs = template_attrs

    print(f"Writing final combined {file_type} data to: {output_path}")
    with ProgressBar():
        combined_ds.to_zarr(output_path, mode='w')

    for ds in ds_list:
        ds.close()
    combined_ds.close()

def combine_forecasts(base_dir):
    """
    Combine forecast files from multiple checkpoints for both atmosphere and ocean.
    Uses checkpointing (via Zarr) so that a job that ends early can be resumed.
    """
    completed_dirs = find_completed_checkpoints(base_dir)
    print(f"Found {len(completed_dirs)} completed checkpoint directories")
    if not completed_dirs:
        raise ValueError("No completed checkpoint directories found")

    # # Process atmosphere files.
    print("\nProcessing atmosphere files...")
    atmos_output = os.path.join(base_dir, "forecast_atmos.zarr")
    process_single_type(completed_dirs, 'atmos', atmos_output)
    gc.collect()

    # Process ocean files.
    print("\nProcessing ocean files...")
    ocean_output = os.path.join(base_dir, "forecast_ocean.zarr")
    process_single_type(completed_dirs, 'ocean', ocean_output)
    gc.collect()

def load_config(config_path):
    """
    Load the JSON configuration file.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine checkpoint ensemble outputs with checkpointing and Zarr format'
    )
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--directory', type=str, help='Output directory path (bypasses config)')

    args = parser.parse_args()

    if not args.config and not args.directory:
        parser.error("You must provide either --config or --directory.")
    if args.config and args.directory:
        parser.error("Please provide only one of --config or --directory, not both.")

    if args.config:
        config = load_config(args.config)
        output_dir = config["output_directory"]
    else:
        output_dir = args.directory

    combine_forecasts(output_dir)
