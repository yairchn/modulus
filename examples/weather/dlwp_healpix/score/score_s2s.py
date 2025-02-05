import os
import gc
import numpy as np
import xarray as xr
import shutil
from postprocessing import (
    calculate_aligned_verification, 
    load_training_data_from_zarr, 
    create_doy_hour,
    )
from score_coupled_forecast import compute_metrics_single_var
from dask.diagnostics import ProgressBar
from collections import defaultdict


def apply_bias_correction(forecast_group, bias):
    """
    Apply bias correction to a subset of the forecast (grouped by year).
    This function iterates over each time step in the group,
    finds the corresponding entry in the bias (by nearest 'doy_hour'),
    aligns the lead times, and subtracts.

    Parameters
    ----------
    forecast_group : xarray.DataArray
        Forecast data for a single year, with dims [time, step, number, ...].
    bias : xarray.DataArray
        Bias data with dims [doy_hour, step, ...].

    Returns
    -------
    xarray.DataArray
        Bias-corrected data for the given group.
    """
    
    forecast_doy_hour = create_doy_hour(forecast_group)

    corrected_data = []
    for i, doy_hour in enumerate(forecast_doy_hour):
        forecast_step = forecast_group.isel(time=i)
        
        matched_bias = bias.sel(doy_hour=doy_hour, method='nearest')
        # consider rechunking 
        matched_bias = matched_bias.chunk({'step': 10})  # example chunk size
        
        # Ensure lead_time alignment
        matched_bias = matched_bias.interp(step=forecast_step.step)
    
        # Check if lead_time coordinates are the same
        if not np.array_equal(forecast_step.step.values, matched_bias.step.values):
            print("Lead times are not aligned!")
        
        # Subtract the bias
        corrected_step = forecast_step - matched_bias
        corrected_data.append(corrected_step)
    
    return xr.concat(corrected_data, dim='time')


def correct_year(year_forecast, bias):
    """
    Correct a single year's forecast.
    """
    corrected_forecast = apply_bias_correction(year_forecast, bias)
    return corrected_forecast


def remove_bias_from_forecast(forecast_da, bias_da):
    """
    Remove bias from forecast DataArray.

    Parameters
    ----------
    forecast_da : xarray.DataArray
        Forecast data with dims [time, step, number, ...].
    bias_da : xarray.DataArray
        Bias data with dims [doy_hour, step, ...].

    Returns
    -------
    xarray.DataArray
        Bias-corrected forecast data.
    """
    if not isinstance(forecast_da, xr.DataArray) or not isinstance(bias_da, xr.DataArray):
        raise ValueError("Both forecast and bias must be xarray DataArrays")

    original_time = forecast_da.time

    grouped = forecast_da.groupby('time.year')
    corrected_forecasts = []
    for year, year_forecast in grouped:
        print(f"Removing bias from year: {year}")
        corrected_year = correct_year(year_forecast, bias_da)
        corrected_forecasts.append(corrected_year)

    corrected_forecast = xr.concat(corrected_forecasts, dim='time')
    corrected_forecast = corrected_forecast.reindex(time=original_time)
    assert np.all(corrected_forecast.time == original_time), "Time dimension mismatch after bias correction"

    return corrected_forecast


def main(
    forecast_path,
    verification_path,
    bias_path,
    variable_names,
    rolling_window=None,
    save_corrected_forecast=False,
    output_dir=None,
    forecast_averaging_window=None
):
    """
    Parameters
    ----------
    forecast_path : str
        Path to original forecast netCDF (with multiple variables).
    verification_path : str (zarr)
        Path to the verification data store (ERA5, etc.).
    bias_path : str
        Path to the bias netCDF, containing the same variables as forecast.
        Dimensions: [doy_hour, step, (ensemble), ...]
    variable_names : list of str
        Which variables to process (e.g. ['z500','t2m','t850','sst']).
    rolling_window : int or None
        If not None, apply a rolling mean over 'doy_hour' dimension of bias for smoothing.
    save_corrected_forecast : bool
        If True, writes the fully corrected forecast to file 
        (bias_corrected_forecast_atmos.nc) for all requested variables.
    output_dir : str or None
        Directory to store the final merged metric files. If None, use `os.path.dirname(forecast_path)`.
    """

    print(f"Loading forecast from {forecast_path} ...")
    forecast_ds = xr.open_dataset(forecast_path)
    if "ckpt_ensemble" in forecast_ds.dims:
        forecast_ds = forecast_ds.rename({"ckpt_ensemble": "ensemble"})

    print(f"Loading bias from {bias_path} ...")
    bias_ds = xr.open_dataset(bias_path)
    if "number" in bias_ds.dims:
        bias_ds = bias_ds.rename({"number": "ensemble"})

    # Optionally smooth bias in doy_hour
    if rolling_window is not None:
        bias_ds = bias_ds.rolling(
            doy_hour=rolling_window, center=True, min_periods=1
        ).mean()

    corrected_forecast_vars = {}

    for var_name in variable_names:
        if var_name not in forecast_ds:
            print(f"Variable '{var_name}' not found in forecast dataset, skipping.")
            continue
        if var_name not in bias_ds:
            print(f"Variable '{var_name}' not found in bias dataset, skipping.")
            continue
        
        print(f"\nBias-correcting variable: {var_name}")
        corrected_var = remove_bias_from_forecast(forecast_ds[var_name], bias_ds[var_name])
        corrected_forecast_vars[var_name] = corrected_var

    # Create a new corrected_forecast Dataset with *only* the corrected variables
    corrected_forecast = xr.Dataset(
        corrected_forecast_vars,
        coords=forecast_ds.coords,
        attrs=forecast_ds.attrs
    )

    # Optionally save the entire corrected forecast
    if save_corrected_forecast:
        out_path = os.path.join(
            os.path.dirname(forecast_path),
            "bias_corrected_forecast_atmos.nc"
        )
        print(f"Saving *all* corrected variables to {out_path} ...")
        corrected_forecast.to_netcdf(out_path)

    print("\nAligning with verification data ...")
    forecast_times = forecast_ds.time.expand_dims(step=forecast_ds.step) + forecast_ds.step
    
    # get the verification for those variables
    forecast_verification_data = load_training_data_from_zarr(
        verification_path,
        forecast_times,
        corrected_forecast_vars
    )

    aligned_verification = calculate_aligned_verification(
        corrected_forecast,  # or forecast_ds if you prefer
        forecast_verification_data
    )

    if forecast_averaging_window is not None:
        # 1) For the bias-corrected forecast
        corrected_forecast = corrected_forecast.coarsen(
            step=forecast_averaging_window, 
            center=True, 
            min_periods=forecast_averaging_window
        ).mean()

        # 2) For the aligned verification
        aligned_verification = aligned_verification.coarsen(
            step=forecast_averaging_window, 
            center=True, 
            min_periods=forecast_averaging_window
        ).mean()

    if output_dir is None:
        output_dir = os.path.dirname(forecast_path)

    evaluation_path = os.path.join(output_dir, "metrics_tmp")
    if os.path.exists(evaluation_path):
        shutil.rmtree(evaluation_path)
    os.makedirs(evaluation_path, exist_ok=True)

    # We'll track which files we produce for each metric
    metric_files = defaultdict(list)

    for var_name in corrected_forecast_vars:
        print(f"\nComputing metrics for bias-corrected variable: {var_name} ...")

        fcst_var = corrected_forecast[var_name]
        verif_var = aligned_verification[var_name]

        with ProgressBar():
            metrics_dict = compute_metrics_single_var(
                verification=verif_var,
                forecast=fcst_var,
                climo_var_da=None,
                spatial_dims=None,
                mask_da=None
            )

        for metric_name, da in metrics_dict.items():
            da.name = metric_name
            out_fn = os.path.join(
                evaluation_path,
                f"{var_name}_{metric_name}.nc"
            )
            print(f"Saving {metric_name} for {var_name} to {out_fn} ...")
            with ProgressBar():
                da.load().to_netcdf(out_fn, mode="w")
            metric_files[metric_name].append(out_fn)

        # Clean up
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

        merged_out = os.path.join(output_dir, f"{metric_name}.nc")
        print(f"  Writing merged dataset to {merged_out}")
        merged_ds.to_netcdf(merged_out, mode="w")
        
        for d in ds_list:
            d.close()

    print("\nFinal metric files created:")
    for metric_name in metric_files.keys():
        merged_out = os.path.join(output_dir, f"{metric_name}.nc")
        print("  ", merged_out)

    print("\nDone!")


if __name__ == "__main__":
    variable_list = ['z500','t2m','t850','sst']
    forecast_averaging_window = 7*4 # average on n step the and verification data before scoring
    forecast_path = "/pscratch/sd/y/yacohen/nvdlesm/inference/HPX64/ensemble_atmos_98_ocean24/forecast_atmos_ensemble.nc"
    bias_path = "/global/cfs/projectdirs/m4935/nvdlesm/ensemble_atmos_98_good_ocean_Jan15_healpix_dir/bias_all_days_z500.nc"
    verification_path = "/global/cfs/projectdirs/m4331/datasets/era5_hpx/era5_hpx64_1980.zarr"
    save_corrected_forecast = False # True to save the bias corrected forecast before applying forecast_averaging_window
    main(
        forecast_path=forecast_path,
        verification_path=verification_path,
        bias_path=bias_path,
        variable_names=variable_list,
        rolling_window=28,
        save_corrected_forecast=save_corrected_forecast,
        output_dir=None
    )
