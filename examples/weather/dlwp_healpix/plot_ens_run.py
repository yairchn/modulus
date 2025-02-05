import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import torch
import earth2grid
import cartopy.crs as ccrs
import pandas as pd


# Load the atmosphere and ocean NetCDF files
atmos_file = '/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/test/ensemble_forecast/forecast_8var_coupled_hpx32_24h_custom-weights_atmos.nc'
ocean_file = '/home/yacohen/mnt/eos/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/test/ensemble_forecast/forecast_hpx32_Coupled2varDLOM_48H_o96H_recunet.nc'

atmos_ds = xr.open_dataset(atmos_file)
ocean_ds = xr.open_dataset(ocean_file)

# Parameters for the HEALPix grid and lat-lon grid
nside = atmos_ds['height'].size
level = int(np.log2(nside))
assert 2**level == nside

# Create the HEALPix grid
data_grid = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)

# Create lat-lon grid
nlat, nlon = 91, 180
lats = np.linspace(-90, 90, nlat)
lons = np.linspace(0, 360, nlon, endpoint=False)
ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)

# Regridder to convert HEALPix to lat-lon
regrid = earth2grid.get_regridder(data_grid, ll_grid)

# Regrid data to lat-lon for a given variable
def regrid_data_to_latlon(variable, ds):
    # Initialize an array to store the regridded data
    regridded_data = np.zeros((ds.sizes['ensemble'], ds.sizes['step'], nlat, nlon))

    # Loop through ensemble and time steps to regrid each slice
    for ens in range(ds.sizes['ensemble']):
        for step in range(ds.sizes['step']):
            # Extract a slice (2D) of the data for this ensemble member and time step
            data_slice = ds[variable].isel(ensemble=ens, step=step).values.flatten()

            # Convert to torch tensor and regrid
            data_tensor = torch.from_numpy(data_slice)
            regrid.to(data_tensor)
            output_data = regrid(data_tensor).numpy()

            # Store the regridded output in the correct location in the array
            regridded_data[ens, step, :, :] = output_data

    return regridded_data


def plot_variable_over_time_multiple_points(variable, ds, points, time_idx, file_name):
    """
    Plots ensemble member trajectories for multiple locations in a 2x2 grid.
    
    Parameters:
    - variable: The variable to plot (e.g., 't2m0', 'sst').
    - ds: The dataset containing the variable.
    - points: A list of tuples [(lat, lon), ...] specifying the locations.
    - time_idx: Time index for the initial time.
    - title: Title prefix for the plots.
    - file_name: File name for saving the figure.
    """
    # Regrid data from HEALPix to lat-lon
    regridded_data = regrid_data_to_latlon(variable, ds)

    # Extract time values and convert to human-readable format
    time_str = pd.to_datetime(ds.time.values[time_idx]).strftime('%Y-%m-%dT%H:%M:%S')
    lead_time = ds['step'].values / np.timedelta64(1, 'h') / 24  # Convert step to days

    # Set up a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()  # Flatten the 2D array of axes

    # Loop over the points (lat, lon) and plot on each subplot
    for i, (lat, lon) in enumerate(points):
        ax = axs[i]

        # Find the closest lat-lon point
        lat_idx = np.abs(lats - lat).argmin()
        lon_idx = np.abs(lons - lon).argmin()

        # Plot for all ensemble members at the selected lat-lon point
        for ens in range(regridded_data.shape[0]):
            if variable == 'sst':
                ax.plot(lead_time, regridded_data[ens, :, lat_idx, lon_idx], label=f'Ensemble {ens+1}')
                ax.set_ylabel(f'{variable}')
            else:
                ax.plot(lead_time, regridded_data[ens, :, lat_idx, lon_idx] - regridded_data[0, :, lat_idx, lon_idx], label=f'Ensemble {ens+1}')
                ax.set_ylabel(f'{variable} perturbed member - control run')

        ax.set_title(f'{lat}°, {lon}°, ({time_str})')
        ax.set_xlabel('Lead Time (days)')
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'./{file_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Define parameters for lat-lon selection
points = [
    (0.0, -120.0),   # Equatorial Eastern Pacific
    (39.7, -105.0),  # Colorado, USA
    (-60.0, 120.0),  # Southern Ocean
    (28.0, 86.9)     # Himalayas, Nepal
]
selected_time = 0  # First time index

selected_variable = 'z500'
plot_variable_over_time_multiple_points(selected_variable, atmos_ds, points, selected_time,
                                       f'{selected_variable}_point_vs_time')

# Plot for t2m0 (atmosphere) at a lat-lon point across time
selected_variable = 't2m0'
plot_variable_over_time_multiple_points(selected_variable, atmos_ds, points, selected_time,
                                       f'{selected_variable}_point_vs_time')

# # Plot for t2m0 (atmosphere) at a lat-lon point across time
# plot_variable_over_time(selected_variable, atmos_ds, selected_lat, selected_lon, selected_time,
#                         f'{selected_variable} for Selected Lat-Lon Point (lat={selected_lat}, lon={selected_lon}) at time ', 
#                         f'{selected_variable}_point_vs_time')


# Plot for sst (ocean) at the same lat-lon point across time
plot_variable_over_time_multiple_points('sst', ocean_ds, points, selected_time,
                                       f'sst_point_vs_time')
# plot_variable_over_time('sst', ocean_ds, selected_lat, selected_lon, selected_time,
#                         f'sst for Selected Lat-Lon Point (lat={selected_lat}, lon={selected_lon}) at time ', 
#                         'sst_point_vs_time')

# 2. Generate contourf maps of ensemble mean and ensemble std for a given lead time

def plot_healpix_regrid(variable, ds, lead_time, title, units, file_name, stats='mean'):
    # Regrid data from HEALPix to lat-lon
    regridded_data = regrid_data_to_latlon(variable, ds)

    # Calculate ensemble mean or std
    if stats == 'mean':
        data = np.mean(regridded_data[:, lead_time, :, :], axis=0)
    else:
        data = np.std(regridded_data[:, lead_time, :, :], axis=0)

    # Create a plot with cartopy for ensemble mean or std
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))

    # Plot ensemble mean using pcolormesh
    max_abs_value = np.max(data)
    min_abs_value = np.min(data)
    im = ax.pcolormesh(lons, lats, np.flipud(data), 
                       transform=ccrs.PlateCarree(), 
                       cmap='viridis', 
                       vmin=min_abs_value, 
                       vmax=max_abs_value)

    # Add coastlines and gridlines
    plt.title(title)
    ax.coastlines()
    ax.gridlines()

    # Add colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.05)
    cbar.set_label(f'{variable} ({units})')

    # Save the figure
    plt.savefig(f'./{file_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Define a lead time for the contourf plots
selected_lead_time = 20 # that is 40days in

# Plot for t2m (atmosphere)
plot_healpix_regrid('t2m0', atmos_ds, selected_lead_time*8, 'Ensemble Mean of t2m0 (Atmosphere)', 'K', 't2m0_mean_map', stats = 'mean')
plot_healpix_regrid('t2m0', atmos_ds, selected_lead_time*8, 'Ensemble STD of t2m0 (Atmosphere)', 'K', 't2m0_std_map', stats = 'std')

plot_healpix_regrid('z500', atmos_ds, selected_lead_time*8, 'Ensemble Mean of z500 (Atmosphere)', '(m/2)^2', 'z500_mean_map', stats = 'mean')
plot_healpix_regrid('z500', atmos_ds, selected_lead_time*8, 'Ensemble STD of z500 (Atmosphere)', '(m/2)^2', 'z500_std_map', stats = 'std')

# Plot for sst (ocean)
plot_healpix_regrid('sst', ocean_ds, selected_lead_time, 'Ensemble Mean of sst (Ocean)', 'K', 'sst_mean_map', stats = 'mean')
plot_healpix_regrid('sst', ocean_ds, selected_lead_time, 'Ensemble STD of sst (Ocean)', 'K', 'sst_std_map', stats = 'std')
