import xarray as xr
import os
import numpy as np
import gc
from memory_profiler import profile
import json 
import argparse

def preprocess_dataset(ds):
    if 'time' in ds.coords:
        if ds['time'].dtype.type is np.string_ or ds['time'].dtype.kind in ['S', 'O', 'U']:
            time_str = ds['time'].astype(str).values
            ds['time'] = np.array(time_str, dtype='datetime64[ns]')

    if 'step' in ds.coords and np.issubdtype(ds['step'].dtype, np.timedelta64):
        ds['step'] = ds['step'].astype('timedelta64[h]').astype(float) / 24.0

    return ds

@profile
def combine_hindcast_files(base_path, zarr_output, filename_pattern, config):
    # Remove existing Zarr store if present to start fresh
    if os.path.exists(zarr_output):
        import shutil
        shutil.rmtree(zarr_output)

    first = True
    for folder_name in sorted(os.listdir(base_path)):
        if folder_name.startswith('hindcast_'):
            file_path = os.path.join(base_path, folder_name, filename_pattern)
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                ds = preprocess_dataset(ds)
                
                if first:
                    # First write: mode='w', no append_dim
                    ds.to_zarr(zarr_output, mode='w', consolidated=True)
                    first = False
                else:
                    # Subsequent writes: mode='a', append_dim='time'
                    ds.to_zarr(zarr_output, mode='a', append_dim='time', consolidated=True)
                
                ds.close()
                gc.collect()
            else:
                print(f"File not found: {file_path}")
    # Reopen the Zarr store to modify global attributes if needed
    with xr.open_zarr(zarr_output, consolidated=True) as combined_ds:
        combined_attrs = combined_ds.attrs.copy()
        for key, value in config.items():
            combined_attrs[key] = value
        
        # Save updated attributes
        combined_ds.attrs.update(combined_attrs)
        combined_ds.close()

def load_config(config_path):
    """
    Load the JSON configuration file.
    Args:
        config_path (str): Path to the JSON configuration file.
    Returns:
        dict: Configuration data as a dictionary.
    """
    with open(config_path, 'r') as file:
        return json.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine checkpoint ensemble outputs')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    base_path = config["base_output_dir"]
    # Parameters
    filename_patterns = ["forecast_atmos.nc","forecast_ocean.nc"]
    for filename_pattern in filename_patterns:
        zarr_output = (f'{base_path}/combined_{filename_pattern[0:-3]}_new_cpu.zarr')
        combine_hindcast_files(base_path, zarr_output, filename_pattern, config)

