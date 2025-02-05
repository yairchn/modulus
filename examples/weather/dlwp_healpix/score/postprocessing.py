import xarray as xr
import pandas as pd
import numpy as np


def is_leap_year(year):
    return ((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)))


def create_doy_hour(da):
    """
    Calculate 'doy_hour' as the cumulative hours since the start of the year
    for each time index in the DataArray.
    """
    doy_hour = ((da.time.dt.dayofyear - 1) * 24 + da.time.dt.hour).astype(np.int32)
    return doy_hour


def load_training_data_from_zarr(zarr_path, forecast_times, variables):
    """
    Load and process training data from a Zarr store.
    This function opens a Zarr dataset, selects specified variables and times,
    and restructures the data into a new xarray Dataset with consistent dimensions.

    Parameters:
    -----------
    zarr_path : str
        Path to the Zarr store containing the dataset.
    forecast_times : numpy.ndarray
        Array of forecast times to select from the dataset.
    variables : list of str
        List of variable names to extract from the dataset.

    Returns:
    --------
    xarray.Dataset
        A new dataset containing the selected variables and times, with dimensions
        ordered as (time, face, height, width) and additional lat/lon coordinates.
    """
    ds = xr.open_zarr(zarr_path)
    if isinstance(forecast_times, np.ndarray):
        unique_times = np.unique(forecast_times.flatten())
    elif isinstance(forecast_times, xr.DataArray):
        unique_times = np.unique(forecast_times.values.flatten())
    else:
        print("Input must be a NumPy array or a xarray DataArray")

    if not isinstance(variables, list):
        variables = [variables]

    data_vars = {}
    for variable in variables:
        variable_data = ds['inputs'].sel(channel_in=variable, time=unique_times)
        data_vars[variable] = variable_data.transpose('time', 'face', 'height', 'width')
    
    new_height_coords = np.arange(data_vars[variables[0]]['height'].size)
    new_width_coords = np.arange(data_vars[variables[0]]['width'].size)

    ds_selected = xr.Dataset(
        data_vars,
        coords={
            'time': variable_data['time'],
            'face': variable_data['face'],
            'height': new_height_coords,
            'width': new_width_coords,
            'lat': (['face', 'height', 'width'], ds['lat'].values),
            'lon': (['face', 'height', 'width'], ds['lon'].values),
        }
    )
    return ds_selected


def calculate_aligned_verification(forecast, verification):
    """
    Align verification data to forecast time and lead_time across all variables.
    
    Parameters:
    forecast (xarray.Dataset): Forecast dataset with time and lead_time dimensions.
    verification (xarray.Dataset): Verification dataset with a time dimension.
    
    Returns:
    xarray.Dataset: Aligned verification dataset.
    """
    verification_times = (forecast.time.expand_dims(step=forecast.step) + forecast.step)
    aligned_verification = verification.sel(time=verification_times)
    aligned_verification['time'] = forecast.time
    return aligned_verification



def create_persistence_forecast(verification_data, variable, forecast_days, freq):
    """
    Generate a persistence forecast where the initial state of the variable is 
    repeated as the forecast for all lead times. The forecast replicates the 
    initial state over the specified forecast range.

    This function is designed to work with Zarr datasets used in NVDLESM and 
    assumes that the time dimension is available and consistent across all forecast 
    lead times. 
    
    Parameters:
    -----------
    verification_data : xarray.Dataset
        The dataset containing the verification data.
    variable : str
        The variable to create the persistence forecast for.
    forecast_days : int
        The number of days for which the forecast should be generated.
    freq : str
        The frequency of the forecast (e.g., '3h' for every 3 hours).
        
    Returns:
    --------
    persistence_forecast : xarray.Dataset
        A dataset containing the persistence forecast with the same spatial 
        dimensions as the input data and a lead_time dimension for forecasts.
    """
    var_data = verification_data[variable]
    forecast_steps = int(forecast_days * 24 / pd.Timedelta(freq).total_seconds() * 3600)
    lead_times = pd.timedelta_range(start='0 days', periods=forecast_steps, freq=freq)
    
    # Truncate the time dimension so that we have enough room for the forecast_steps
    forecast_times = var_data.time[:len(var_data.time) - forecast_steps + 1]
    forecast_array = np.empty((len(forecast_times), forecast_steps, len(var_data.face), len(var_data.height), len(var_data.width)))
    
    for i, time in enumerate(forecast_times):
        initial_value = var_data.sel(time=time).values
        # Tile the initial_value to replicate across the lead_time dimension
        forecast_array[i, :, :, :, :] = np.tile(initial_value, (forecast_steps, 1, 1, 1))
    
    persistence_forecast = xr.Dataset(
        {
            variable: (('time', 'lead_time', 'face', 'height', 'width'), forecast_array)
        },
        coords={
            'time': forecast_times,
            'lead_time': lead_times,
            'face': var_data.face,
            'height': var_data.height,
            'width': var_data.width,
            'lat': (['face', 'height', 'width'], var_data.lat.values),
            'lon': (['face', 'height', 'width'], var_data.lon.values),
        }
    )
    return persistence_forecast


def persistence_forecast_from_time_arrays(verification_data, variable, time_dataset, lead_time_dataset):
    """
    Wrapper function to create a persistence forecast based on time and lead time datasets.
    
    Parameters:
    -----------
    verification_data : xarray.Dataset
        The dataset containing the verification data.
    variable : str
        The variable to create the persistence forecast for.
    time_dataset : xarray.DataArray or pandas.DatetimeIndex
        Dataset or Index containing the forecast initialization times.
    lead_time_dataset : xarray.DataArray or pandas.TimedeltaIndex
        Dataset or Index containing the forecast lead times.
    
    Returns:
    --------
    persistence_forecast : xarray.Dataset
        A dataset containing the persistence forecast.
    """
    # Convert datasets to pandas objects if they're xarray DataArrays
    if isinstance(time_dataset, xr.DataArray):
        time_dataset = time_dataset.to_index()
    if isinstance(lead_time_dataset, xr.DataArray):
        lead_time_dataset = lead_time_dataset.to_index()
    
    var_data = verification_data[variable]
    
    # Create an empty array to store the persistence forecast
    forecast_array = np.empty((len(time_dataset), len(lead_time_dataset), len(var_data.face), len(var_data.height), len(var_data.width)))
    
    # Generate the persistence forecast
    for i, time in enumerate(time_dataset):
        if time in var_data.time:
            initial_value = var_data.sel(time=time).values
        else:
            # If the exact time is not in var_data, use the nearest available time
            initial_value = var_data.sel(time=time, method="nearest").values
        
        # Tile the initial_value to replicate across the lead_time dimension
        forecast_array[i, :, :, :, :] = np.tile(initial_value, (len(lead_time_dataset), 1, 1, 1))
    
    # Create the xarray Dataset
    persistence_forecast = xr.Dataset(
        {
            variable: (('time', 'lead_time', 'face', 'height', 'width'), forecast_array)
        },
        coords={
            'time': time_dataset,
            'lead_time': lead_time_dataset,
            'face': var_data.face,
            'height': var_data.height,
            'width': var_data.width,
            'lat': (['face', 'height', 'width'], var_data.lat.values),
            'lon': (['face', 'height', 'width'], var_data.lon.values),
        }
    )
    
    return persistence_forecast


def create_lagged_ensemble_forecast(
    deterministic_forecast: xr.Dataset,
    variable: str,
    n_ensemble: int
) -> xr.Dataset:
    """
    Create a lagged ensemble probabilistic forecast using a deterministic forecast dataset.

    Parameters:
    -----------
    deterministic_forecast : xr.Dataset
        The deterministic forecast data containing the variable to be used.
        
    variable : str
        The name of the variable to use for generating the ensemble forecast.
        
    n_ensemble : int
        The number of ensemble members (must be an odd number).
        
    Returns:
    --------
    ensemble_forecast : xr.Dataset
        The lagged ensemble forecast with dimensions ['time', 'lead_time', 'ensemble', 'face', 'height', 'width'].
    """
    if n_ensemble % 2 == 0:
        raise ValueError("n_ensemble must be an odd number.")
    
    half_window = n_ensemble // 2
    forecast_times = deterministic_forecast['time']
    lead_times = deterministic_forecast['lead_time']
    
    time_step = (forecast_times[1] - forecast_times[0]).values
    valid_times = forecast_times
    
    ensemble_members = []
    
    for lag in range(-half_window, half_window + 1):
        shifted_times = forecast_times + lag * time_step
        
        # Handle out-of-bounds by extrapolating with NaN
        lagged_forecast = deterministic_forecast[variable].reindex(time=shifted_times, method="nearest", tolerance=pd.Timedelta('1s'), fill_value=np.nan)
        lagged_forecast = lagged_forecast.assign_coords(time=valid_times)
        ensemble_members.append(lagged_forecast)
    
    ensemble_forecast = xr.concat(ensemble_members, dim='ensemble')

    ensemble_indices = np.arange(n_ensemble)
    ensemble_forecast = ensemble_forecast.assign_coords(ensemble=ensemble_indices)
    
    ensemble_dataset = xr.Dataset(
        {variable: ensemble_forecast},
        coords={
            'time': valid_times,
            'lead_time': lead_times,
            'ensemble': ensemble_indices,
            'face': deterministic_forecast['face'],
            'height': deterministic_forecast['height'],
            'width': deterministic_forecast['width'],
            'lat': deterministic_forecast['lat'],
            'lon': deterministic_forecast['lon']
        }
    )
    
    # Preserve the attributes of the original dataset
    ensemble_dataset.attrs = deterministic_forecast.attrs
    return ensemble_dataset
