import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import earth2grid
import torch
import cartopy.crs as ccrs

# Load the data
bias = xr.open_dataset('/pscratch/sd/y/yacohen/nvdlesm/hindcast_output/hpx64_deterministic/bias_t2m.nc')
variable = 't2m'
units = 'K'
nside = bias.height.size
level = int(np.log2(nside))
assert 2**level == nside

# Create the HEALPix grid
data_grid = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
week = 5
leadtime_slice = slice(7*(week-1)*4, 7*week*4)
# Extract the data
data = bias[variable].sel(statistic="mean").mean(dim="time").isel(lead_time=leadtime_slice).mean(dim="lead_time").values.reshape((-1,))
data_tensor = torch.from_numpy(data)

# Create lat-lon grid
nlat, nlon = 91, 180
lats = np.linspace(-90, 90, nlat)
lons = np.linspace(0, 360, nlon, endpoint=False)
ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)

# Regrid the data
regrid = earth2grid.get_regridder(data_grid, ll_grid)
regrid.to(data_tensor)
output_data = regrid(data_tensor).numpy()

# Create a plot with cartopy
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))

# Plot the data
max_abs_value = np.max(np.abs(output_data))

# Create the pcolormesh with a centered colormap
im = ax.pcolormesh(lons, lats, np.flipud(output_data), 
                   transform=ccrs.PlateCarree(), 
                   cmap='seismic', 
                   vmin=-max_abs_value, 
                   vmax=max_abs_value)

# Add coastlines and gridlines
plt.title("mean bias")
ax.coastlines()
ax.gridlines()

# Add colorbar
cbar = plt.colorbar(im, orientation='horizontal', pad=0.05)
cbar.set_label(f'{variable} Bias ({units})')

# Save the figure
plt.savefig(f'./{variable}_mean_bias_map_week{week}.png', dpi=300, bbox_inches='tight')
plt.close()


# data = bias[variable].sel(statistic="std", time=bias['time'].dt.month.isin([6,7,8])).mean(dim="time").isel(lead_time=slice(84,112)).mean(dim="lead_time").values.reshape((-1,))
data = bias[variable].sel(statistic="std").mean(dim="time").isel(lead_time=leadtime_slice).mean(dim="lead_time").values.reshape((-1,))
data_tensor = torch.from_numpy(data)

# Create lat-lon grid
nlat, nlon = 91, 180
lats = np.linspace(-90, 90, nlat)
lons = np.linspace(0, 360, nlon, endpoint=False)
ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)

# Regrid the data
regrid = earth2grid.get_regridder(data_grid, ll_grid)
regrid.to(data_tensor)
output_data = regrid(data_tensor).numpy()

# Create a plot with cartopy
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))

# Plot the data
max_abs_value = np.max(np.abs(output_data))

# Create the pcolormesh with a centered colormap
im = ax.pcolormesh(lons, lats, np.flipud(output_data), 
                   transform=ccrs.PlateCarree(), 
                   cmap='viridis', 
                   vmin=0, 
                   vmax=max_abs_value)

# Add coastlines and gridlines
plt.title("std bias")
ax.coastlines()
ax.gridlines()

# Add colorbar
cbar = plt.colorbar(im, orientation='horizontal', pad=0.05)
cbar.set_label(f'{variable} Bias ({units})')

# Save the figure
plt.savefig(f'./{variable}_std_bias_map_week{week}.png', dpi=300, bbox_inches='tight')
plt.close()