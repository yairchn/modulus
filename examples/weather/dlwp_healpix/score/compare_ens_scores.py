import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# Optional metadata for label units or scale factors
variable_metas = {
    't850': {'scale_factor': 1.0, 'units': '°C'},
    'z500': {'scale_factor': 1 / 9.81, 'units': 'm'},
    # Add more as needed...
}

def plot_rmse(ax, rmse_file, ens_rmse_files, variable, spatial_dims=None,
              ens_rmse_labels=None, ens_rmse_colors=None):
    """
    Plot RMSE lines on ax:
     - Plot RMSE from a single file in black with low opacity.
     - Plot ensemble RMSE from multiple files in different colors.
    
    Parameters:
      - ax: Matplotlib Axes object.
      - rmse_file: Path to the primary RMSE NetCDF file.
      - ens_rmse_files: List of paths to ensemble RMSE NetCDF files.
      - variable: Variable name to plot.
      - spatial_dims: Dimensions to average over.
      - ens_rmse_labels: List of labels for ensemble RMSE files.
      - ens_rmse_colors: List of colors for ensemble RMSE files.
    """
    if spatial_dims is None:
        spatial_dims = ['time']
    
    if not os.path.isfile(rmse_file):
        print(f"File not found: {rmse_file}")
        return
    
    _, ext = os.path.splitext(rmse_file)
    if ext == '.zarr':
        ds = xr.open_dataset(rmse_file, engine='zarr')
    elif ext == '.nc':
        ds = xr.open_dataset(rmse_file)
        
    if variable not in ds:
        print(f"Variable '{variable}' not found in {rmse_file}.")
        ds.close()
        return

    data = ds[variable]
    if 'step' not in data.dims:
        print(f"No 'step' dimension in {rmse_file}, skipping.")
        ds.close()
        return

    sf = variable_metas.get(variable, {}).get('scale_factor', 1.0)
    data = data.mean(spatial_dims)
    x_vals = data.step / np.timedelta64(1, 'D')
    y_vals = data.values * sf
    ax.plot(x_vals, y_vals, color='black', alpha=0.2, label='RMSE')
    ds.close()
    
    # Plot ensemble RMSE from the list of files
    num_ens = len(ens_rmse_files)
    if ens_rmse_colors is None:
        ens_rmse_colors = plt.cm.viridis(np.linspace(0, 1, num_ens))
    if ens_rmse_labels is None:
        ens_rmse_labels = [f'Ensemble RMSE {i+1}' for i in range(num_ens)]
    
    for ens_file, label, color in zip(ens_rmse_files, ens_rmse_labels, ens_rmse_colors):
        if not os.path.isfile(ens_file):
            print(f"File not found: {ens_file}")
            continue

        _, ext_ens = os.path.splitext(ens_file)
        if ext_ens == '.zarr':
            ds_ens = xr.open_dataset(ens_file, engine='zarr')
        elif ext_ens == '.nc':
            ds_ens = xr.open_dataset(ens_file)
        if variable not in ds_ens:
            print(f"Variable '{variable}' not found in {ens_file}.")
            ds_ens.close()
            continue
        
        data_ens = ds_ens[variable].mean(spatial_dims)
        x_vals = data_ens.step / np.timedelta64(1, 'D')
        y_vals = data_ens.values * sf
        ax.plot(x_vals, y_vals, color=color, label=label, alpha=0.8)
        ds_ens.close()

def plot_crps(ax, crps_files, variable, spatial_dims=None,
              crps_labels=None, crps_colors=None):
    """
    Plot CRPS lines on ax:
     - Plot CRPS from multiple files in different colors.
    
    Parameters:
      - ax: Matplotlib Axes object.
      - crps_files: List of paths to CRPS NetCDF files.
      - variable: Variable name to plot.
      - spatial_dims: Dimensions to average over.
      - crps_labels: List of labels for CRPS files.
      - crps_colors: List of colors for CRPS files.
    """
    if spatial_dims is None:
        spatial_dims = ['time']
    
    num_crps = len(crps_files)
    if crps_colors is None:
        crps_colors = plt.cm.plasma(np.linspace(0, 1, num_crps))
    if crps_labels is None:
        crps_labels = [f'CRPS {i+1}' for i in range(num_crps)]
    
    for crps_file, label, color in zip(crps_files, crps_labels, crps_colors):
        if not os.path.isfile(crps_file):
            print(f"File not found: {crps_file}")
            continue
        _, ext = os.path.splitext(crps_file)
        if ext == '.zarr':
            ds = xr.open_dataset(crps_file, engine='zarr')
        elif ext == '.nc':
            ds = xr.open_dataset(crps_file)
        if variable not in ds:
            print(f"Variable '{variable}' not found in {crps_file}.")
            ds.close()
            continue
        
        data = ds[variable].mean(spatial_dims)
        x_vals = data.step / np.timedelta64(1, 'D')
        sf = variable_metas.get(variable, {}).get('scale_factor', 1.0)
        y_vals = data.values * sf
        ax.plot(x_vals, y_vals, color=color, label=label)
        ds.close()

def plot_std_list(ax, std_files, variable, spatial_dims=None,
                  std_labels=None, std_colors=None):
    """
    Plot STD lines on ax from a list of files:
     - For each std file, plot the standard deviation.
     - If an ensemble dimension exists, the ensemble mean is plotted.
    
    Parameters:
      - ax: Matplotlib Axes object.
      - std_files: List of paths to STD NetCDF files.
      - variable: Variable name to plot.
      - spatial_dims: Dimensions to average over.
      - std_labels: List of labels for STD files.
      - std_colors: List of colors for STD files.
    """
    if spatial_dims is None:
        spatial_dims = ['time']
    
    num_std = len(std_files)
    if std_colors is None:
        std_colors = plt.cm.cividis(np.linspace(0, 1, num_std))
    if std_labels is None:
        std_labels = [f'STD {i+1}' for i in range(num_std)]
    
    for std_file, label, color in zip(std_files, std_labels, std_colors):
        if not os.path.isfile(std_file):
            print(f"File not found: {std_file}")
            continue

        _, ext = os.path.splitext(std_file)
        if ext == '.zarr':
            ds_std = xr.open_dataset(std_file, engine='zarr')
        elif ext == '.nc':
            ds_std = xr.open_dataset(std_file)
        if variable not in ds_std:
            print(f"Variable '{variable}' not found in {std_file}.")
            ds_std.close()
            continue
        
        data_std = ds_std[variable]
        if 'step' not in data_std.dims:
            print(f"No 'step' dimension in {std_file}, skipping.")
            ds_std.close()
            continue
        
        sf = variable_metas.get(variable, {}).get('scale_factor', 1.0)
        data_std = data_std.mean(spatial_dims)
        # If an ensemble dimension exists, compute the ensemble mean
        if 'ensemble' in data_std.dims:
            data_std = data_std.mean(dim='ensemble')
        
        x_vals = data_std.step / np.timedelta64(1, 'D')
        y_vals = data_std.values * sf
        ax.plot(x_vals, y_vals, color=color, label=label, alpha=0.8, linestyle='--')
        ds_std.close()

def plot_all_metrics(rmse_file, ens_rmse_files, crps_files, std_files, variable='t850',
                     output_file='metrics_plot.png',
                     ens_rmse_labels=None, ens_rmse_colors=None,
                     crps_labels=None, crps_colors=None,
                     std_labels=None, std_colors=None,
                     spatial_dims=None,
                     ifs_file=None):
    """
    Create a figure with two panels:
      - Left panel: CRPS (from multiple files) plus optional IFS CRPS baseline.
      - Right panel: RMSE (from a single file plus ensemble members) and STD (from multiple files)
        plus optional IFS ensemble-mean RMSE baseline.
    
    Parameters:
      - rmse_file: Path to the primary RMSE NetCDF file.
      - ens_rmse_files: List of paths to ensemble RMSE NetCDF files.
      - crps_files: List of paths to CRPS NetCDF files.
      - std_files: List of paths to STD NetCDF files.
      - variable: Variable name to plot.
      - output_file: Filename for the saved plot.
      - ens_rmse_labels: List of labels for ensemble RMSE files.
      - ens_rmse_colors: List of colors for ensemble RMSE files.
      - crps_labels: List of labels for CRPS files.
      - crps_colors: List of colors for CRPS files.
      - std_labels: List of labels for STD files.
      - std_colors: List of colors for STD files.
      - spatial_dims: Dimensions to average over.
      - ifs_file: Optional path to a file containing IFS baseline metrics.
    """
    if spatial_dims is None:
        spatial_dims = ['time']
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left panel: CRPS ---
    ax_crps = axs[0]
    plot_crps(ax_crps, crps_files, variable,
              spatial_dims=spatial_dims,
              crps_labels=crps_labels,
              crps_colors=crps_colors)
    # Plot IFS baseline for CRPS if provided
    if ifs_file is not None and os.path.isfile(ifs_file):
        ds_ifs = xr.open_dataset(ifs_file)
        if 'crps_bias_corrected' in ds_ifs:
            arr_ifs = ds_ifs['crps_bias_corrected']
            if 'time' in arr_ifs.dims:
                arr_ifs = arr_ifs.mean('time')
            if 'step' in arr_ifs.dims:
                x_vals = arr_ifs.step / np.timedelta64(1, 'D')
                y_vals = arr_ifs.values 
                ax_crps.plot(x_vals, y_vals, color='gray', linewidth=3, label='IFS (bias-corr)')
        ds_ifs.close()
    
    ax_crps.set_title('CRPS')
    ax_crps.set_xlabel('Lead Time (days)')
    ax_crps.set_ylabel(f'{variable} [{variable_metas.get(variable, {}).get("units", "")}]')
    ax_crps.grid(True)
    ax_crps.legend()

    # --- Right panel: RMSE & STD ---
    ax_rmse = axs[1]
    plot_rmse(ax_rmse, rmse_file, ens_rmse_files, variable,
              spatial_dims=spatial_dims,
              ens_rmse_labels=ens_rmse_labels,
              ens_rmse_colors=ens_rmse_colors)
    plot_std_list(ax_rmse, std_files, variable,
                  spatial_dims=spatial_dims,
                  std_labels=std_labels,
                  std_colors=std_colors)
    # Plot IFS baseline for RMSE if provided
    if ifs_file is not None and os.path.isfile(ifs_file):
        ds_ifs = xr.open_dataset(ifs_file)
        if 'ens_rmse_bias_corrected' in ds_ifs:
            arr_ifs = ds_ifs['ens_rmse_bias_corrected']
            if 'time' in arr_ifs.dims:
                arr_ifs = arr_ifs.mean('time')
            if 'step' in arr_ifs.dims:
                x_vals = arr_ifs.step / np.timedelta64(1, 'D')
                # ifs z500 is already in m
                y_vals = arr_ifs.values 
                ax_rmse.plot(x_vals, y_vals, color='gray', linewidth=3, label='IFS ens-mean (bias-corr)')
        ds_ifs.close()
    
    ax_rmse.set_title('RMSE & Spread (STD)')
    ax_rmse.set_xlabel('Lead Time (days)')
    ax_rmse.set_ylabel(f'{variable} [{variable_metas.get(variable, {}).get("units", "")}]')
    ax_rmse.grid(True)
    ax_rmse.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    print(f"Saved figure to {output_file}")

if __name__ == "__main__":
    # Example usage:
    folder = "/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_1atmos/"
    folder3 = "/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_1atmos_bv_10step/"
    folder4 = "/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_1atmos_bv_4step/"
    folder5 = "/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_1atmos_bv_1step/"
    folder6 = "/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_8atmos_bv_1step/"
    
    # RMSE file (single file)
    rmse_file = f"{folder}rmse.nc"
    
    # Ensemble RMSE files (multiple)
    ens_rmse_files = [
        f"{folder}ens_rmse.nc",
        f"{folder3}ens_rmse.nc",
        f"{folder4}ens_rmse.nc",
        f"{folder5}ens_rmse.nc",
        f"{folder6}ens_rmse.nc"
    ]
    ens_rmse_labels = [
        "eRMSE 6",
        "eRMSE 6x1x3, BV 10",
        "eRMSE 6x1x3, BV 4",
        "eRMSE 6x1x3, BV 1",
        "eRMSE 6x8x3, BV 1",
    ]
    ens_rmse_colors = [
        "navy",
        "crimson",
        "darkorange",
        "darkgreen",
        "deeppink"
    ]
    
    # CRPS files (multiple)
    crps_files = [
        f"{folder}crps.nc",
        f"{folder3}crps.nc",
        f"{folder4}crps.nc",
        f"{folder5}crps.nc",
        f"{folder6}crps.nc"
    ]
    crps_labels = [
        "6",
        "6x1x3, BV 10",
        "6x1x3, BV 4",
        "6x1x3, BV 1",
        "6x8x3, BV 1"
    ]
    crps_colors = [
        "navy",
        "crimson",
        "darkorange",
        "darkgreen",
        "deeppink"
    ]
    
    # STD files (multiple)
    std_files = [
        f"{folder}std_dev.nc",
        f"{folder3}std_dev.nc",
        f"{folder4}std_dev.nc",
        f"{folder5}std_dev.nc",
        f"{folder6}std_dev.nc"
    ]
    std_labels = [
        "STD 6x1x3 ",
        "STD 6x1x3, BV 10",
        "STD 6x1x3, BV 4",
        "STD 6x1x3, BV 1",
        "STD 6x8x3, BV 1"
    ]
    std_colors = [
        "navy",
        "crimson",
        "darkorange",
        "darkgreen",
        "deeppink"
    ]
    
    # Variable to plot
    variable = 't850'
    
    # Output plot file
    output_file = f"{variable}_ensemble_rmse_crps_std_plot_ext_new.png"
    
    # Optional IFS file for baseline metrics (update this path as needed)
    ifs_file = f"/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/pharrington/datasets/ecmwf_s2s/error_bias/scores_by_day_{variable}.nc"
    
    plot_all_metrics(
        rmse_file=rmse_file,
        ens_rmse_files=ens_rmse_files,
        crps_files=crps_files,
        std_files=std_files,
        variable=variable,
        output_file=output_file,
        ens_rmse_labels=ens_rmse_labels,
        ens_rmse_colors=ens_rmse_colors,
        crps_labels=crps_labels,
        crps_colors=crps_colors,
        std_labels=std_labels,
        std_colors=std_colors,
        ifs_file=ifs_file
    )
