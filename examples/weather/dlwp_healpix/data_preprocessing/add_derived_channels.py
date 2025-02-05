import zarr
import numpy as np
import argparse
import os
from glob import glob
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
import earth2grid
import numpy as np
import torch
from scipy import interpolate

class interpolate_sst(torch.nn.Module):
    def __init__(self, width, nlat=360, nlon=720) -> None:
        super().__init__()
        # Create the healpix and latlon regridders and save them for later
        self.width = width
        self.nlat = nlat
        self.nlon = nlon
        self.level = int(np.log2(width))
        self.hpx = earth2grid.healpix.Grid(
            level=self.level,
            pixel_order=earth2grid.healpix.HEALPIX_PAD_XY,
        )
        self.lats = np.linspace(-90, 90, nlat)
        self.lons = np.linspace(-180, 180, nlon, endpoint=False)
        self.ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
        self.regrid = earth2grid.get_regridder(self.hpx, self.ll_grid)
        self.regrid_back = earth2grid.get_regridder(self.ll_grid, self.hpx)


    def forward(self, sst, lsm): 
        # Regrid sst and lsm to regular grid
        data_flat_sst = torch.from_numpy(sst.flatten()).to(torch.float64)
        data_flat_lsm = torch.from_numpy(lsm.flatten()).to(torch.float64)

        sst_regridded = self.regrid(data_flat_sst).reshape(self.nlat, self.nlon).numpy()
        lsm_regridded = self.regrid(data_flat_lsm).reshape(self.nlat, self.nlon).numpy()

        # Initialize sst_interp with NaNs
        sst_interp = np.full_like(sst_regridded, np.nan)

        # Perform zonal interpolation
        for i in range(self.nlat):
            sst_row = sst_regridded[i, :]
            lsm_row = lsm_regridded[i, :]
            lon_row = self.lons  # from -180 to 180 degrees

            # Find ocean points
            ocean_mask = (lsm_row == 0) & (~np.isnan(sst_row))
            ocean_lons = lon_row[ocean_mask]
            ocean_sst = sst_row[ocean_mask]

            if len(ocean_sst) > 1:
                # Wrap the data around the -180/180 discontinuity
                # Shift longitudes to a 0 to 360 range for wrapping
                ocean_lons_360 = np.mod(ocean_lons + 360, 360)
                lon_row_360 = np.mod(lon_row + 360, 360)

                # Extend the data for wrapping
                extended_ocean_lons = np.concatenate([
                    ocean_lons_360 - 360,
                    ocean_lons_360,
                    ocean_lons_360 + 360
                ])
                extended_ocean_sst = np.tile(ocean_sst, 3)

                # Sort the extended longitudes and corresponding SST
                sorted_indices = np.argsort(extended_ocean_lons)
                extended_ocean_lons_sorted = extended_ocean_lons[sorted_indices]
                extended_ocean_sst_sorted = extended_ocean_sst[sorted_indices]

                # Interpolate in the 0-360 range
                f = interpolate.interp1d(
                    extended_ocean_lons_sorted, extended_ocean_sst_sorted, kind='linear',
                    bounds_error=False, fill_value=np.nan
                )
                sst_interp_row = f(lon_row_360)
                sst_interp[i, :] = sst_interp_row
            elif len(ocean_sst) == 1:
                # Use the single ocean value for the entire row
                sst_interp[i, :] = ocean_sst[0]
            else:
                # No ocean points, leave as NaN
                pass

        # Identify the southernmost latitude index with any valid SST values
        valid_lat_mask = ~np.isnan(sst_interp).all(axis=1)
        valid_lat_indices = np.where(valid_lat_mask)[0]
        
        if len(valid_lat_indices) > 0:
            southernmost_valid_lat_index = valid_lat_indices[-1]  # Last index with valid data

            # Extract SST values at the southernmost valid latitude
            southernmost_valid_sst = sst_interp[southernmost_valid_lat_index, :]

            # Fill NaN values south of this latitude
            lat_indices_to_fill = np.arange(southernmost_valid_lat_index + 1, self.nlat)
            # latitudes_to_fill = lats[lat_indices_to_fill]

            if len(lat_indices_to_fill) > 0:
                for i in lat_indices_to_fill:
                    sst_interp[i, :] = southernmost_valid_sst
            else:
                print("No latitudes to fill south of the southernmost valid latitude.")
        else:
            print("Warning: No valid SST data found to fill NaN values south of the southernmost valid latitude.")

        # Regrid back to HEALPIX grid
        data_interp_tensor = torch.from_numpy(sst_interp).to(torch.float64)
        #regrid_back = earth2grid.get_regridder(ll_grid, hpx)
        sst_interp_hpx_flat = self.regrid_back(data_interp_tensor)
        sst_interp_hpx_ = sst_interp_hpx_flat.reshape(12, 2**self.level, 2**self.level).numpy()
        # verify that values over the ocean are unchanged
        sst_interp_hpx = np.copy(sst_interp_hpx_)
        ocean_mask = ~np.isnan(sst)
        sst_interp_hpx[ocean_mask] = sst[ocean_mask]

        return sst_interp_hpx


def calculate_tau(z300, z700):
    """Calculate geopotential thickness (tau) between 300 and 700 hPa levels."""
    return z300 - z700


def calculate_windspeed(u, v):
    """Calculate windspeed from u and v components."""
    return np.sqrt(u**2 + v**2)


def add_q2m(sp, t2m):
    """Calculate 2-metre specific humidity from surface pressure and t2m."""
    q2m = specific_humidity_from_dewpoint(sp * units.Pa, t2m * units.degK)
    return q2m.magnitude


def add_derived_channels(file_path):
    print(f"Processing {file_path}")
    ds = zarr.open(file_path, mode='r+')
    
    # Get the existing channel names
    channel_in = ds.channel_in[:]
    channel_out = ds.channel_out[:]
    channel_c = ds.channel_c[:]
    
    # Add tau300-700
    if 'tau300-700' not in channel_in and 'z300' in channel_in and 'z700' in channel_in:
        print("Adding tau300-700")
        z300_index = np.where(channel_in == 'z300')[0][0]
        z700_index = np.where(channel_in == 'z700')[0][0]
        
        tau = calculate_tau(ds.inputs[:, z300_index], ds.inputs[:, z700_index])
        
        # Resize inputs and targets to accommodate the new channel
        new_shape = list(ds.inputs.shape)
        new_shape[1] += 1
        ds.inputs.resize(*new_shape)
        ds.targets.resize(*new_shape)
        
        # Add tau to inputs and targets
        ds.inputs[:, -1] = tau
        ds.targets[:, -1] = tau
        
        # Update channel lists
        channel_in = np.append(channel_in, 'tau300-700')
        channel_out = np.append(channel_out, 'tau300-700')
    elif 'z300' not in channel_in or 'z700' not in channel_in:
        print("Missing data needed to calculate tau300-700, skipping")
      
    # Add windspeed at various levels
    for level in ['10m', '1000', '850', '700', '500', '250', '100', '50']:
        windspeed_name = f'ws{level}'
        if windspeed_name not in channel_in and f'u{level}' in channel_in and f'v{level}' in channel_in:
            print(f"Adding {windspeed_name}")
            u_index = np.where(channel_in == f'u{level}')[0][0]
            v_index = np.where(channel_in == f'v{level}')[0][0]
            
            windspeed = calculate_windspeed(ds.inputs[:, u_index], ds.inputs[:, v_index])
            
            # Resize inputs and targets to accommodate the new channel
            new_shape = list(ds.inputs.shape)
            new_shape[1] += 1
            ds.inputs.resize(*new_shape)
            ds.targets.resize(*new_shape)
            
            # Add windspeed to inputs and targets
            ds.inputs[:, -1] = windspeed
            ds.targets[:, -1] = windspeed
            
            # Update channel lists
            channel_in = np.append(channel_in, windspeed_name)
            channel_out = np.append(channel_out, windspeed_name)
    
    # Add q2m
    if 'q2m' not in channel_in and 'sp' in channel_in and 't2m' in channel_in:
        print("Adding q2m")
        sp_index = np.where(channel_in == 'sp')[0][0]
        t2m_index = np.where(channel_in == 't2m')[0][0]
        
        sp_data = ds.inputs[:, sp_index]
        t2m_data = ds.inputs[:, t2m_index]
        
        q2m = add_q2m(sp_data, t2m_data)
        
        # Resize inputs and targets to accommodate the new channel
        new_shape = list(ds.inputs.shape)
        new_shape[1] += 1
        ds.inputs.resize(*new_shape)
        ds.targets.resize(*new_shape)
        
        # Add q2m to inputs and targets
        ds.inputs[:, -1] = q2m
        ds.targets[:, -1] = q2m
        
        # Update channel lists
        channel_in = np.append(channel_in, 'q2m')
        channel_out = np.append(channel_out, 'q2m')
    
    # interpolate sst over land
    # two possible LSM channels
    if 'lsm' in channel_c:
        lsm_index = np.where(channel_c == 'lsm')[0][0]
    elif 'land_sea_mask' in channel_c:
        lsm_index = np.where(channel_c == 'land_sea_mask')[0][0]
    else:
        lsm_index = -1 # simpler to test for -1 than None

    if 'sst' in channel_in and lsm_index >= 0:
        print("Interpolating sst over land")
        sst_index = np.where(channel_in == 'sst')[0][0]
        lsm_data = ds.constants[lsm_index]
        lsm_data = np.nan_to_num(lsm_data, nan=0.0)

        interp = interpolate_sst(width=lsm_data.shape[-1], nlat=360, nlon=720)

        num_tsteps = ds.inputs.shape[0]
        for i in range(num_tsteps):
            sst_data = ds.inputs[i, sst_index]
            ds.inputs[i, sst_index] = interp(sst_data, lsm_data)
        
    # Update channel_in and channel_out datasets
    ds.channel_in.resize(len(channel_in))
    ds.channel_in[:] = channel_in
    ds.channel_out.resize(len(channel_out))
    ds.channel_out[:] = channel_out
    
    # Consolidate metadata
    zarr.consolidate_metadata(file_path)
    print(f"Finished processing {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Add derived channels to yearly Zarr files.")
    parser.add_argument("--target_folder", type=str, required=True, help="Path to the folder containing yearly Zarr files")
    args = parser.parse_args()

    # Get all Zarr files in the target folder
    zarr_files = sorted(glob(os.path.join(args.target_folder, "*.zarr")))

    for zarr_file in zarr_files:
        add_derived_channels(zarr_file)

if __name__ == "__main__":
    main()
