import os
import gc
import shutil
import xarray as xr
import numpy as np
import xskillscore as xs
from dask.diagnostics import ProgressBar
from collections import defaultdict
import dask
from postprocessing import calculate_aligned_verification, load_training_data_from_zarr


def unify_ensemble_dims(ds):
    """
    Unify ensemble dimensions in an xarray Dataset.

    This function will:
      - Find all dimensions whose names contain 'ensemble'.
      - If a dimension is exactly 'ensemble', rename it to a temporary name.
      - Chunk these ensemble dimensions with a chunk size of -1.
      - Stack them into a new dimension called 'ensemble'.
      - Reset the new 'ensemble' index so that the stacked dimensions become regular coordinates.

    Parameters:
      ds (xarray.Dataset): The input dataset with ensemble dimensions.

    Returns:
      xarray.Dataset: The dataset with ensemble dimensions unified into a single dimension.
    """
    # Identify all dims that have 'ensemble' in their name.
    ensemble_dims = [dim for dim in ds.dims if "ensemble" in dim]

    # If one of these dims is exactly 'ensemble', rename it to a temporary name.
    rename_map = {}
    for dim in ensemble_dims:
        if dim == "ensemble":
            rename_map[dim] = "temp_ens"
    if rename_map:
        ds = ds.rename(rename_map)
        # Update the list of ensemble dimensions to reflect any renaming.
        ensemble_dims = [rename_map.get(dim, dim) for dim in ensemble_dims]

    # Chunk all ensemble dimensions with a chunk size of -1.
    chunk_dict = {dim: -1 for dim in ensemble_dims}
    ds = ds.chunk(chunk_dict)

    # Stack all ensemble-related dimensions into a new dimension called 'ensemble'.
    ds = ds.stack(ensemble=ensemble_dims)
    ds = ds.reset_index("ensemble")
    
    return ds


def compute_acc(
    verification,
    forecast,
    climo_var_da,
    mask_da=None,
    spatial_dims=None
):
    """
    Computes ACC, optionally with spatial weighting if mask_da is provided.
    
    Parameters
    ----------
    verification : xr.DataArray
        Verification data (time, step, face, height, width[, ensemble])
    forecast : xr.DataArray
        Forecast data with the same dims as verification (+/- ensemble).
    climo_var_da : xr.DataArray
        Climatology array (dayofyear, face, height, width).
    mask_da : xr.DataArray, optional
        Fractional weighting array (face, height, width). If provided, 
        we do weighted means; otherwise unweighted.
    spatial_dims : list of str, optional
        Spatial dims to average over. Default ["face", "height", "width"].
    
    Returns
    -------
    xr.DataArray
        ACC vs. lead time step (and ensemble if present). 
        Dims => (step,) or (ensemble, step).
    """
    if spatial_dims is None:
        spatial_dims = ["face", "height", "width"]

    climo = xr.full_like(forecast, fill_value=np.nan)

    for tval in forecast["time"].values:
        for sval in forecast["step"].values:
            date_daily = (tval + sval).astype("datetime64[D]")
            day_of_year = (date_daily - date_daily.astype("datetime64[Y]")).astype(int) + 1

            slice_climo = climo_var_da.sel(dayofyear=day_of_year)
            climo.loc[{"time": tval, "step": sval}] = slice_climo


    f_anom = forecast - climo
    v_anom = verification - climo

    dims_to_average = spatial_dims + ["time"] 

    if mask_da is not None:
        numerator = (f_anom * v_anom).weighted(mask_da).mean(dim=dims_to_average, skipna=True)
        denom_f   = (f_anom**2).weighted(mask_da).mean(dim=dims_to_average, skipna=True)
        denom_v   = (v_anom**2).weighted(mask_da).mean(dim=dims_to_average, skipna=True)
    else:
        # Unweighted
        numerator = (f_anom * v_anom).mean(dim=dims_to_average, skipna=True)
        denom_f   = (f_anom**2).mean(dim=dims_to_average, skipna=True)
        denom_v   = (v_anom**2).mean(dim=dims_to_average, skipna=True)

    acc_arr = numerator / np.sqrt(denom_f * denom_v)
    acc_arr.name = "acc"

    return acc_arr.astype("float32")


def compute_metrics_single_var(
    verification, 
    forecast, 
    climo_var_da=None, 
    spatial_dims=None,
    mask_da=None
):
    """
    Compute RMSE, CRPS, std dev, ensemble RMSE, and ACC (if climo provided).
    If mask_da is provided, we do weighted metrics.
    
    Parameters
    ----------
    verification : xr.DataArray
    forecast : xr.DataArray
    climo_var_da : xr.DataArray, optional
    spatial_dims : list of str, optional
    mask_da : xr.DataArray, optional
        Weights for xskillscore calls (0..1).
    
    Returns
    -------
    metrics_dict : dict
        with keys in ["ens_rmse", "std_dev", "rmse", "crps", "acc"(opt)]
    """
    if spatial_dims is None:
        spatial_dims = ["face", "height", "width"]

    verif_chunk = {"step": 1, "height": 64, "width": 64}
    fcst_chunk_crps = {"ensemble": -1, "step": 1, "height": 64, "width": 64}
    fcst_chunk_rmse = {"ensemble": 1,  "step": 1, "height": 64, "width": 64}

    print("compute_metrics_single_var - crps")
    crps = xs.crps_ensemble(
        verification.chunk(verif_chunk),
        forecast.chunk(fcst_chunk_crps),
        member_dim="ensemble",
        dim=spatial_dims,
        weights=mask_da
    )
    print("compute_metrics_single_var - std")
    std_per_ens = forecast.chunk(fcst_chunk_crps).std("ensemble")
    if mask_da is not None:
        std_dev = std_per_ens.weighted(mask_da).mean(spatial_dims)
    else:
        std_dev = std_per_ens.mean(spatial_dims)

    print("compute_metrics_single_var - ens_rmse")
    ens_rmse = xs.rmse(
        verification,
        forecast.chunk(fcst_chunk_crps).mean("ensemble"),
        dim=spatial_dims,
        weights=mask_da,
        skipna=True
    )
    print("compute_metrics_single_var - rmse")
    rmse = xs.rmse(
        verification.chunk(verif_chunk),
        forecast.chunk(fcst_chunk_rmse),
        dim=spatial_dims,
        weights=mask_da,
        skipna=True
    )

    crps    = crps.astype("float32")
    std_dev = std_dev.astype("float32")
    ens_rmse= ens_rmse.astype("float32")
    rmse    = rmse.astype("float32")

    metrics_dict = {
        "ens_rmse": ens_rmse,
        "std_dev":  std_dev,
        "rmse":     rmse,
        "crps":     crps
    }

    if climo_var_da is not None:
        acc_da = compute_acc(
            verification.chunk(verif_chunk),
            forecast.chunk(fcst_chunk_rmse),
            climo_var_da,
            mask_da=mask_da,
            spatial_dims=spatial_dims
        )
        metrics_dict["acc"] = acc_da

    return metrics_dict


def main(
    atmos_forecast_path,
    ocean_forecast_path,
    verification_path,
    climatology_path=None
):
    """
    Read atmosphere variables from 'atmos_forecast_path', 'sst' from 'ocean_forecast_path',
    then compute metrics and save them as merged files (e.g. rmse.nc with z500,t2m,t850,sst).
    """
    forecast_atmos = xr.open_zarr(atmos_forecast_path)
    forecast_ocean = xr.open_zarr(ocean_forecast_path)

    forecast_atmos = unify_ensemble_dims(forecast_atmos)
    forecast_ocean = unify_ensemble_dims(forecast_ocean)
    
    save_directory = os.path.dirname(atmos_forecast_path)
    evaluation_path = os.path.join(save_directory, "metrics_tmp")
    if os.path.exists(evaluation_path):
        shutil.rmtree(evaluation_path)
    os.makedirs(evaluation_path, exist_ok=True)
    
    variable_names_atmos = ["z500" , "t850"]
    variable_names_total = variable_names_atmos + ["sst"]

    forecast_times = forecast_atmos.time.expand_dims(step=forecast_atmos.step) + forecast_atmos.step

    forecast_verification_data = load_training_data_from_zarr(
        verification_path,
        forecast_times,
        variable_names_total
    )
    
    atmos_aligned_verification = calculate_aligned_verification(
        forecast_atmos, 
        forecast_verification_data
    )
    ocean_aligned_verification = calculate_aligned_verification(
        forecast_ocean, 
        forecast_verification_data
    )
    

    if climatology_path is not None:
        ds_climo = xr.open_dataset(climatology_path)
    else:
        ds_climo = None

    land_sea_da = xr.open_dataset(verification_path, engine='zarr')\
                    .constants.sel(channel_c='land_sea_mask')\
                    .squeeze(drop=True)
    mask_da = (1 - land_sea_da).astype("float32") 
    mask_da = xr.where(mask_da < 1, 0, mask_da)

    metric_files = defaultdict(list)
    
    for var_name in variable_names_total:
        print(f"\nProcessing variable: {var_name} ...")

        if var_name in variable_names_atmos:
            fcst_var = forecast_atmos[var_name]
            verif_var = atmos_aligned_verification[var_name]
        else:
            fcst_var = forecast_ocean[var_name]
            verif_var = ocean_aligned_verification[var_name]

        if var_name == "sst":
            var_mask_da = mask_da
        else:
            var_mask_da = None

        if ds_climo is not None and "targets" in ds_climo.data_vars:
            climo_var_da = ds_climo["targets"].sel(channel_out=var_name)
        else:
            climo_var_da = None

        with ProgressBar():
            metrics_dict = compute_metrics_single_var(
                verif_var,
                fcst_var,
                climo_var_da=climo_var_da,
                mask_da=var_mask_da
            )

        for metric_name, da in metrics_dict.items():
            da.name = metric_name
            out_fn = os.path.join(evaluation_path, f"{var_name}_{metric_name}_sst.nc")
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                with ProgressBar():
                    da.load().to_netcdf(out_fn, mode="w")
            metric_files[metric_name].append(out_fn)

        del fcst_var, verif_var, metrics_dict
        gc.collect()
    
    print("\nAll metrics computed and saved to disk in 'metrics_tmp' folder.")

    print("\nCombining metric files by metric type (crps, rmse, etc.) ...")
    for metric_name, files in metric_files.items():
        print(f"  Merging files for metric = {metric_name}:")
        ds_list = []
        for fpath in files:
            print(f"    {fpath}")
            var_n = os.path.basename(fpath).split("_")[0]
            tmp_ds = xr.open_dataset(fpath).rename({metric_name: var_n})
            ds_list.append(tmp_ds)
        
        merged_ds = xr.merge(ds_list)
        
        merged_out = os.path.join(save_directory, f"{metric_name}.nc")
        print(f"  Writing merged dataset to {merged_out}")
        merged_ds.to_netcdf(merged_out, mode="w")
        
        for d in ds_list:
            d.close()

    print("\nDone!")


if __name__ == "__main__":
    verification_path = '/lustre/fsw/coreai_climate_earth2/datasets/healpix/era5_hpx64_1980.zarr'
    atmos_forecast_path = "/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_8atmos_bv_1step/forecast_atmos.zarr"
    ocean_forecast_path = "/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/good_6ocean_8atmos_bv_1step/forecast_ocean.zarr"
    climatology_path = "/lustre/fsw/coreai_climate_earth2/datasets/healpix/climatology_era5_hpx64_1980_jan31.nc"
    main(
        atmos_forecast_path,
        ocean_forecast_path,
        verification_path,
        climatology_path
    )