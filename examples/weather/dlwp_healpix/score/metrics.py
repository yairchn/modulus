import numpy as np
import xarray as xr
import pandas as pd


def weighted_mean(data, lat=None):
    """
    Calculate the Mean weighted by latitude if provided.
    
    Parameters:
    - data: 3D or 4D xr.DataArray (time, face, height, width)
    - lat: latitude points that correspond to data (3D or 4D with face, height, and width)
    
    Returns: float
        Value of area-weighted mean
    """
    if lat is None:
        return data.mean(dim=('face', 'width', 'height'))
    else:
        weights = xr.DataArray(np.cos(np.deg2rad(lat.values)), dims=('face', 'height', 'width'))
        weights /= weights.mean()
        return data.weighted(weights).mean(dim=('face', 'width', 'height'))
    

def pointwise_error(forecast, verification):
    """
    Calculate pointwise error between forecast and verification data with exact time matching using xr.apply_ufunc.
    
    Parameters:
    forecast (xr.Dataset): Forecast dataset
    verification (xr.Dataset): Verification dataset
    
    Returns:
    xr.DataArray: error values with 'time' and 'lead_time' as dimensions
    Now handles lat and lon dimensions.
    """
    error = xr.apply_ufunc(
        np.subtract,
        forecast,
        verification,
        dask="allowed"
    )
    
    return error

def weighted_error(forecast, verification, lat=None):
    """
    Calculate weighted error between forecast and verification data with exact time matching using xr.apply_ufunc.
    
    Parameters:
    forecast (xr.Dataset): Forecast dataset
    verification (xr.Dataset): Verification dataset
    variable (str): Variable name to compute error for
    lat (xr.DataArray, optional): Latitude points to use for weighting
    
    Returns:
    xr.DataArray: Weighted error values with 'time' and 'lead_time' as dimensions
    """
    error = pointwise_error(forecast, verification)
    if lat is not None:
        error = weighted_mean(error, lat)
    return error


def weighted_ensemble_std(data, lat=None):
    """
    Calculate the area weighted mean of the standard deviation along the ensemble dimension of a DataArray.

    Parameters:
    - data: xarray DataArray, the data for which to calculate the standard deviation
    - lat_dim: string, the name of the latitude dimension in `data`

    Returns:
    - Weighted standard deviation of `data`
    """
    return weighted_mean(data.std("ensemble"), lat)

    
def weighted_mse(pred, target, lat=None):
    """
    Calculate the Mean Square Error, weighted by latitude if provided.
    
    Parameters:
    - pred, target: 2D xarray DataArrays, predicted values and corresponding targets for a given channel
    - lat: latitude points that correspond to pred and target
    
    Returns: float
        Value of area-weighted MSE
    """
    squared_error = xr.apply_ufunc(np.square, pointwise_error(pred, target), dask="allowed")
    mse = weighted_mean(squared_error, lat)
    return mse


def spread_skill_ratio(pred, target, dim="latlon", lat=None):
    """
    Calculate the spread-skill ratio, either by averaging over time or space, 
    optionally weighted by latitude.
    
    Parameters:
    - pred: xarray.DataArray
        Multi-dimensional array of ensemble forecast data with dimensions 
        ['time', 'ensemble', 'height', 'width'].
    - target: xarray.DataArray
        Multi-dimensional array of target data with dimensions 
        ['time', 'height', 'width'].
    - dim: str, optional, default="latlon"
        If "time", calculate the mean over the time dimension. 
        If "latlon", calculate the mean over the spatial dimensions, 
        weighted by latitude if provided.
    - lat: xarray.DataArray, optional
        Latitude points corresponding to the forecast and target data 
        for area weighting. Only used if dims="latlon".
    
    Returns: float
        The area- or time-weighted spread-skill ratio.
    """
    squared_error = xr.apply_ufunc(np.square, pointwise_error(pred.mean(dim="ensemble"), target), dask="allowed")
    
    if dim == "time":
        skill = np.sqrt(squared_error.mean(dim="time"))
        spread = np.sqrt(pred.var("ensemble").mean(dim="time"))
    else:
        skill = np.sqrt(weighted_mean(squared_error, lat))
        spread = np.sqrt(weighted_mean(pred.var("ensemble"), lat))
    
    n_members = pred.sizes["ensemble"]
    spread_skill_ratio = spread / skill * np.sqrt((n_members + 1) / n_members)
    
    return spread_skill_ratio


def weighted_mae(pred, target, lat=None):
    """
    Calculate the Mean Absolute Error, weighted by latitude if provided.
    
    Parameters:
    - pred, target: 2D xarray DataArrays representing predicted and true values for a given channel.
    - lat: latitude points that correspond to pred and target (optional).
    
    Returns: float
        Value of the area-weighted MAE.
    """
    abs_error = xr.apply_ufunc(np.abs, pointwise_error(pred, target), dask="allowed")
    mae = weighted_mean(abs_error, lat)
    return mae


def threshold_brier_score(pred, target, threshold=5):
    """
    Calculate the threshold-based Brier score for probabilistic forecasts.

    Parameters:
    pred : xr.DataArray
        Probabilistic forecast of a variable, which must include an ensemble dimension.
    target : xr.DataArray
        Observation of a variable with dimensions (lat, lon).
    threshold : float, optional, default=5
        Value that turns the continuous observation into a binary event.

    Returns:
    xr.DataArray
        Map of Brier score values.
    """
    
    target_binary = (target > threshold).astype(int)
    pred_binary = (pred > threshold).mean(dim="ensemble")
    brier_score = (pred_binary - target_binary) ** 2
    return brier_score


def crps_from_empirical_cdf_xarray(truth: xr.DataArray, ensemble: xr.DataArray) -> xr.DataArray:
    """Compute the exact CRPS using the CDF method with xarray DataArrays

    Args:
        truth: (...) DataArray of observations
        ensemble: (N, ...) DataArray of ensemble members

    Returns:
        (...,) DataArray of CRPS scores
    """
    y = truth
    n = ensemble.sizes['ensemble']
    ensemble_sorted = ensemble.sortby('ensemble')

    ans = xr.zeros_like(truth)

    # dx [F(x) - H(x-y)]^2 = dx [0 - 1]^2 = dx
    val = ensemble_sorted.isel(ensemble=0) - y
    ans += xr.where(val > 0, val, 0.0)

    for i in range(n - 1):
        x0 = ensemble_sorted.isel(ensemble=i)
        x1 = ensemble_sorted.isel(ensemble=i + 1)

        cdf = (i + 1) / n

        # a. case y < x0
        val = (x1 - x0) * (cdf - 1) ** 2
        mask = y < x0
        ans += xr.where(mask, val, 0.0)

        # b. case x0 <= y <= x1
        val = (y - x0) * cdf**2 + (x1 - y) * (cdf - 1) ** 2
        mask = (y >= x0) & (y <= x1)
        ans += xr.where(mask, val, 0.0)

        # c. case x1 < y
        mask = y > x1
        val = (x1 - x0) * cdf**2
        ans += xr.where(mask, val, 0.0)

    # dx [F(x) - H(x-y)]^2 = dx [1 - 0]^2 = dx
    val = y - ensemble_sorted.isel(ensemble=-1)
    ans += xr.where(val > 0, val, 0.0)
    return ans


def crps(pred, target):
    """
    Compute CRPS for each variable in the given Datasets of truth and ensemble predictions.

    Args:
        truth_ds: xr.Dataset containing truth values for one or more variables.
        ensemble_ds: xr.Dataset containing ensemble predictions for the same variables.

    Returns:
        xr.Dataset containing CRPS values for each variable.
    """
    if not any(['ensemble' in pred[var_name].dims for var_name in pred.data_vars]):
        raise ValueError("The 'ensemble' dimension is missing from the pred Dataset.")

    crps_results = {}
    for var_name in target.data_vars:
        truth_var = target[var_name]
        ensemble_var = pred[var_name]
        crps_results[var_name] = crps_from_empirical_cdf_xarray(truth_var, ensemble_var)
    return xr.Dataset(crps_results)

