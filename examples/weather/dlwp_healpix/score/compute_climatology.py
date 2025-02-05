import os
import xarray as xr


verification_path = '/global/cfs/projectdirs/m4331/datasets/era5_hpx/era5_hpx64_1980.zarr'
dataset_name = os.path.basename(verification_path).split('.')[0]

ds_verif = xr.open_zarr(verification_path)

climatology_all_channels = (
    ds_verif['targets']
    .groupby('time.dayofyear')
    .mean(dim='time')
    .compute()
)

climatology_all_channels = climatology_all_channels.chunk({
    "dayofyear": 1,
    "channel_out": 1,
    "face": 12,
    "height": 64,
    "width": 64
})

climatology_ds = climatology_all_channels.to_dataset(name="climatology")
out_file = os.path.join(
    os.path.dirname(verification_path),
    f"climatology_with_rmse_{dataset_name}.nc"
)
ds_verif.close()

climatology_ds.to_zarr(
    out_file,
    mode="w",
    consolidated=True
)
print(f"Saved final dataset (climatology + RMSE) to {out_file}")

climatology_path = os.path.join(
    os.path.dirname(verification_path),
    f"climatology_{dataset_name}.nc"
)
print("before saving")
climatology_all_channels.to_netcdf(climatology_path)
print(f"Climatology saved to {climatology_path}")
