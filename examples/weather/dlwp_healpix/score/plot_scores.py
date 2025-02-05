import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os    

# Optional metadata for label units or scale factors
variable_metas = {
    't850': {'scale_factor': 1.0, 'units': 'C'},
    'z500': {'scale_factor': 1 / 9.81, 'units': 'm'},
    'sst':  {'scale_factor': 1.0, 'units': 'C'},
    # Add more as needed...
}


def plot_acc(ax, acc_file, variable, spatial_dims=None):
    """
    Plot ACC lines on ax:
    - If 'ensemble' is present, plot each ensemble member (ens=0 in blue, others in black w/ alpha),
      and ensemble-mean in red (or adapt as needed).
    - If no 'ensemble', single line in blue.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    acc_file : str
        Path to the ACC netCDF file.
    variable : str
        Variable to plot.  (Must be a key in acc_file.)
    spatial_dims : list, optional
        Dimensions over which to average before plotting. Defaults to ['time'].
    """
    
    if spatial_dims is None:
        spatial_dims = ['time']

    if not os.path.isfile(acc_file):
        print(f"File not found: {acc_file}")
        return

    ds = xr.open_dataset(acc_file)

    if variable not in ds:
        print(f"Variable '{variable}' not found in {acc_file}.")
        ds.close()
        return

    data = ds[variable]
    if 'step' not in data.dims:
        print(f"No 'step' dimension in {acc_file}, skipping.")
        ds.close()
        return

    # If there's an ensemble dimension, plot them individually + their mean
    if 'ensemble' in data.dims:
        for i, ens_val in enumerate(data.ensemble.values):
            arr = data.sel(ensemble=ens_val)
            x_vals = arr.step / np.timedelta64(1, 'D')
            y_vals = arr.values
            if i == 0:
                ax.plot(x_vals, y_vals, color='blue', label='ACC (ens=0)')
            else:
                ax.plot(x_vals, y_vals, color='black', alpha=0.2)

        # Also plot overall ensemble mean
        arr_mean = data.mean('ensemble')
        x_vals = arr_mean.step / np.timedelta64(1, 'D')
        y_vals = arr_mean.values
        ax.plot(x_vals, y_vals, color='red', label='ACC ens-mean')

    else:
        # Single-member scenario
        arr = data
        x_vals = arr.step / np.timedelta64(1, 'D')
        y_vals = arr.values
        ax.plot(x_vals, y_vals, color='blue', label='ACC')

    ds.close()


def plot_rmse(ax, rmse_file, ens_rmse_file, variable, spatial_dims=None, apply_scale=True):
    """
    Plot RMSE lines on ax:
     - If 'ensemble' is present:
       - plot each ensemble member (ens=0 in blue, others in black w/ alpha)
       - also plot ensemble-mean RMSE in red
     - If no 'ensemble', plot a single line in blue

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    rmse_file : str
        Path to the RMSE netCDF file (possibly single-member).
    ens_rmse_file : str
        Path to the ensemble-mean netCDF file.
    variable : str
        Variable to plot.
    spatial_dims : list, optional
        Dimensions over which to average before plotting. Defaults to ['time'].
    apply_scale : bool
        Whether to apply scale factor to the data or not.
    """
    if spatial_dims is None:
        spatial_dims = ['time']

    if not os.path.isfile(rmse_file):
        print(f"File not found: {rmse_file}")
        return
    if not os.path.isfile(ens_rmse_file):
        print(f"File not found: {ens_rmse_file}")
        return

    ds = xr.open_dataset(rmse_file)
    ds_ens = xr.open_dataset(ens_rmse_file)

    if variable not in ds:
        print(f"Variable '{variable}' not found in {rmse_file}.")
        ds.close()
        ds_ens.close()
        return

    data = ds[variable]
    data_ens = ds_ens[variable]
    if 'step' not in data.dims:
        print(f"No 'step' dimension in {rmse_file}, skipping.")
        ds.close()
        ds_ens.close()
        return

    # Retrieve scale factor from metadata only if apply_scale=True
    sf = 1.0
    if apply_scale:
        sf = variable_metas.get(variable, {}).get('scale_factor', 1.0)

    # Average over spatial_dims first
    data = data.mean(spatial_dims)

    # Check for ensemble dimension
    if 'ensemble' in data.dims:
        # Plot each ensemble member
        for i, ens_val in enumerate(data.ensemble.values):
            arr = data.sel(ensemble=ens_val)
            x_vals = arr.step / np.timedelta64(1, 'D')
            y_vals = arr.values * sf

            if i == 0:
                ax.plot(x_vals, y_vals, color='blue', label='RMSE (ens=0)')
            else:
                ax.plot(x_vals, y_vals, color='black', alpha=0.2)

        # Plot ensemble mean RMSE across ensemble
        arr_mean = data_ens.mean(spatial_dims)
        x_vals = arr_mean.step / np.timedelta64(1, 'D')
        y_vals = arr_mean.values * sf
        ax.plot(x_vals, y_vals, color='red', label='RMSE ens-mean')
    else:
        # Single line (no ensemble dimension)
        arr = data
        x_vals = arr.step / np.timedelta64(1, 'D')
        y_vals = arr.values * sf
        ax.plot(x_vals, y_vals, color='blue', label='RMSE')

    ds.close()
    ds_ens.close()

def plot_std(ax, std_file, variable, spatial_dims=None, apply_scale=True):
    """
    Plot STD lines on ax:
     - If 'ensemble' is present, plot each ensemble member in the same color
       or style. (Here we simply plot a single line or ensemble-mean if needed.)
     - If no 'ensemble', single line.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    std_file : str
        Path to the standard deviation netCDF file.
    variable : str
        Variable to plot.
    spatial_dims : list, optional
        Dimensions over which to average before plotting. Defaults to ['time'].
    apply_scale : bool
        Whether to apply scale factor to the data or not.
    """
    if spatial_dims is None:
        spatial_dims = ['time']

    if not os.path.isfile(std_file):
        print(f"File not found: {std_file}")
        return

    ds = xr.open_dataset(std_file)
    if variable not in ds:
        print(f"Variable '{variable}' not found in {std_file}.")
        ds.close()
        return

    data = ds[variable]
    if 'step' not in data.dims:
        print(f"No 'step' dimension in {std_file}, skipping.")
        ds.close()
        return

    sf = 1.0
    if apply_scale:
        sf = variable_metas.get(variable, {}).get('scale_factor', 1.0)

    data = data.mean(spatial_dims)
    x_vals = data.step / np.timedelta64(1, 'D')
    y_vals = data.values * sf
    ax.plot(x_vals, y_vals, color='orange', label='STD')

    ds.close()

def plot_crps(ax, crps_file, variable, spatial_dims=None, apply_scale=True):
    """
    Plot CRPS lines on ax:
    - If 'ensemble' is present, plot each ensemble member or adapt as needed.
    - If no 'ensemble', single line.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    crps_file : str
        Path to the CRPS netCDF file.
    variable : str
        Variable to plot.
    spatial_dims : list, optional
        Dimensions over which to average before plotting. Defaults to ['time'].
    apply_scale : bool
        Whether to apply scale factor to the data or not.
    """
    if spatial_dims is None:
        spatial_dims = ['time']

    if not os.path.isfile(crps_file):
        print(f"File not found: {crps_file}")
        return

    ds = xr.open_dataset(crps_file)
    if variable not in ds:
        print(f"Variable '{variable}' not found in {crps_file}.")
        ds.close()
        return

    data = ds[variable]
    if 'step' not in data.dims:
        print(f"No 'step' dimension in {crps_file}, skipping.")
        ds.close()
        return

    sf = 1.0
    if apply_scale:
        sf = variable_metas.get(variable, {}).get('scale_factor', 1.0)

    data = data.mean(spatial_dims)
    x_vals = data.step / np.timedelta64(1, 'D')
    y_vals = data.values * sf
    ax.plot(x_vals, y_vals, color='green', label='CRPS')

    ds.close()

def plot_all_metrics(
    rmse_file, 
    ens_rmse_file, 
    std_file, 
    crps_file, 
    acc_file,
    variable='t850', 
    output_file='metrics_plot.png',
    ifs_file=None
):
    """
    Create a figure with three panels:
      - Left panel: CRPS (with optional IFS baseline if available)
      - Middle panel: RMSE + STD (with optional IFS baseline if available)
      - Right panel: ACC (with optional IFS baseline if available)

    If 'ifs_file' is provided and exists, we also plot IFS baselines (no scaling).
    If 'acc_file' is provided, plot forecast ACC as a third panel. 
    """
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    import os

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    # ------------------------------------------------------------------ #
    # 1) Left panel: CRPS
    ax_crps = axs[0]
    plot_crps(ax_crps, crps_file, variable, apply_scale=True)

    # If we have an IFS file for baseline CRPS
    if ifs_file is not None and os.path.isfile(ifs_file):
        ds_ifs = xr.open_dataset(ifs_file)
        if 'crps_bias_corrected' in ds_ifs:
            arr_ifs = ds_ifs['crps_bias_corrected']
            if 'step' in arr_ifs.dims:
                arr_ifs = arr_ifs.mean('time')
                x_vals = arr_ifs.step / np.timedelta64(1, 'D')
                y_vals = arr_ifs.values
                ax_crps.plot(x_vals, y_vals, color='magenta', label='IFS CRPS (bias-corr)')
        ds_ifs.close()

    ax_crps.set_title('CRPS')
    ax_crps.set_xlabel('Lead Time (days)')
    ax_crps.set_ylabel(f'{variable} [{variable_metas.get(variable,{}).get("units","")}]')
    ax_crps.grid(True)
    ax_crps.legend()
    ax_crps.set_ylim([0,45])
    ax_crps.set_xlim([0,45])

    # ------------------------------------------------------------------ #
    # 2) Middle panel: RMSE & STD
    ax_rmse_std = axs[1]
    plot_rmse(ax_rmse_std, rmse_file, ens_rmse_file, variable, apply_scale=True)
    plot_std(ax_rmse_std, std_file, variable, apply_scale=True)

    # If we have an IFS file for baseline RMSE
    if ifs_file is not None and os.path.isfile(ifs_file):
        ds_ifs = xr.open_dataset(ifs_file)
        if 'ens_rmse_bias_corrected' in ds_ifs:
            arr_ifs = ds_ifs['ens_rmse_bias_corrected']
            if 'step' in arr_ifs.dims:
                arr_ifs = arr_ifs.mean('time')
                x_vals = arr_ifs.step / np.timedelta64(1, 'D')
                y_vals = arr_ifs.values
                ax_rmse_std.plot(x_vals, y_vals, color='darkviolet', label='IFS ens-mean (bias-corr)')
        ds_ifs.close()

    ax_rmse_std.set_title('RMSE & Spread')
    ax_rmse_std.set_xlabel('Lead Time (days)')
    ax_rmse_std.set_ylabel(f'{variable} [{variable_metas.get(variable,{}).get("units","")}]')
    ax_rmse_std.grid(True)
    ax_rmse_std.legend()
    ax_rmse_std.set_ylim([0,130])
    ax_rmse_std.set_xlim([0,45])

    # ------------------------------------------------------------------ #
    # 3) Right panel: ACC
    ax_acc = axs[2]
    
    # Plot your forecast ACC if provided
    if acc_file is not None and os.path.isfile(acc_file):
        plot_acc(ax_acc, acc_file, variable)
    
    # If the IFS file is provided, try to plot baseline ACC
    if ifs_file is not None and os.path.isfile(ifs_file):
        ds_ifs = xr.open_dataset(ifs_file)
        # You can choose between 'acc_bias_corrected' or 'acc_ens_mean_bias_corrected'
        # Here we show an example with 'acc_ens_mean_bias_corrected' 
        # (which is presumably the ensemble mean ACC).
        if 'acc_ens_mean_bias_corrected' in ds_ifs:
            arr_ifs = ds_ifs['acc_ens_mean_bias_corrected']
            if 'step' in arr_ifs.dims:
                arr_ifs = arr_ifs.mean('time')
                x_vals = arr_ifs.step / np.timedelta64(1, 'D')
                y_vals = arr_ifs.values
                ax_acc.plot(x_vals, y_vals, color='darkorange', label='IFS ACC (bias-corr)')
        ds_ifs.close()

    ax_acc.set_title('ACC')
    ax_acc.set_xlabel('Lead Time (days)')
    ax_acc.set_ylabel('ACC (unitless)')
    ax_acc.grid(True)
    ax_acc.legend()
    ax_acc.set_xlim([0,45])

    # ------------------------------------------------------------------ #
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    print(f"Saved figure to {output_file}")


if __name__ == "__main__":
    # Example usage:
    folder = "/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_8atmos_bv_1step/"
    folder2 = "/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_8atmos/"
    rmse_file = f"{folder2}/rmse_prunned.nc"
    ens_rmse_file = f"{folder}/ens_rmse.nc"
    std_file = f"{folder}/std_dev.nc"
    crps_file = f"{folder}/crps.nc"
    acc_file = f"{folder2}/acc_prunned.nc"
    variable = 'z500'
    ifs_file = f"/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/pharrington/datasets/ecmwf_s2s/error_bias/scores_by_day_{variable}.nc"

    plot_all_metrics(
        rmse_file, 
        ens_rmse_file, 
        std_file, 
        crps_file,
        acc_file,
        variable=variable,
        output_file=f"./{variable}_metrics_good_6ocean_8atmos_bv1_test.png",
        ifs_file=ifs_file  # Pass the optional baseline file here
    )