# imports 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import logging 
import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('log')

# library with variable specific constants
# IMPORTANT: This dictionary should be updated with the scale_factor and units for each new variable used in the analysis
variable_metas = {
    'z500': {
        'scale_factor':1/9.81, # Scale factor to convert source units to desired plotting physical units, e.g. 1/9.81 for geopotential height to meters
        'units':'m',           # Physical units of the variable to be used in RMSE label, e.g. 'm' for geopotential height
    },
    'z1000': {
        'scale_factor':1/9.81,
        'units':'m',
    },
    'z250': {
        'scale_factor':1/9.81,
        'units':'m',
    },
    't850': {
        'scale_factor':1,
        'units':'C',
    },
    't2m0': {
        'scale_factor':1,
        'units':'C',
    },
}

def format_verif(verif, forecast, variable):
    """
    Takes a dataset in the DLWP-style zarr format (channel_c, channel_in, channel_out, face, heigh, width, time) and 
    extracts desired variable and converts to format useful for analysis (face, height, width, time, step)

    Parameters:
    verif (xarray.Dataset): The verification data in DLWP-style zarr format.
    forecast (xarray.Dataset): The forecast data in DLWP-output-style netcdf format.
    variable (str): The variable to extract from the verification data.

    Returns:
    xarray.DataArray: The verification data in aligned with forecast.
    """

    # extract variable from verification data
    verif = verif['targets'].sel(channel_out=variable).squeeze()

    # align verification data with forecast data
    verif_aligned = xr.full_like(forecast, fill_value=np.nan)
    for t in forecast.time:
        for s in forecast.step:
            verif_aligned.loc[{'time': t, 'step': s}] = verif.sel(time=t + s).values

    return verif_aligned

def rmse(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
        return_rmse=False,
): 
    """
    Calculates the Root Mean Square Error (RMSE) for a set of forecasts and plots and/or the results.

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to be used for that forecast. 
        'rmse_cache' (str, optional): The file path where the calculated RMSE should be cached. If None, the RMSE is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the RMSE is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to None.
    return_rmse (bool, optional): If True, the function returns the calculated RMSE. Defaults to False.

    Returns:
    list of xarray.DataArray: A list of RMSE values for each forecast. Only returned if return_rmse is True.
    """
    rmse = []
    # iterate through forecats and obtain rmse 
    for forecast_param in forecast_params:
        if os.path.isfile(forecast_param.get('rmse_cache','')):
            logger.info(f"Loading RMSE from {forecast_param['rmse_cache']}.")
            rmse.append(xr.open_dataarray(forecast_param['rmse_cache']))
        else: 
            logger.info(f"Calculating RMSE for {forecast_param['file']}.")
        
            # open forecast and verification data Yair
            forecast = xr.open_dataset(forecast_param['file'])[variable]
            forecast = forecast.isel(ensemble=0)
            verif = format_verif(xr.open_dataset(forecast_param['verification_file'], engine='zarr'),
                                 forecast,
                                 variable)

            # calculate rmse 
            rmse.append(np.sqrt(((forecast - verif) ** 2).mean(dim=['time', 'face', 'height','width'])))

            # cache rmse if requested
            if forecast_param.get('rmse_cache',None) is not None:
                logger.info(f"Caching RMSE to {forecast_param['rmse_cache']}.")
                # create directory if it doesn't exist
                os.makedirs(os.path.dirname(forecast_param['rmse_cache']), exist_ok=True)
                rmse[-1].to_netcdf(forecast_param['rmse_cache'])

    # plot RMSE if requested
    if plot_file is not None:
        fig, ax = plt.subplots()
        for skill, plot_kwargs in zip(rmse, [forecast_param['plot_kwargs'] for forecast_param in forecast_params]):
            ax.plot([s / np.timedelta64(1, 'D') for s in skill.step.values], # plot in days 
                    skill * variable_metas[variable]['scale_factor'], # scale to physical units
                    **plot_kwargs) # style curve and label

        # style plot
        ax.set_xlabel('Forecast Days')
        ax.set_ylabel(f'RMSE [{variable_metas[variable]["units"]}]')
        # calculate y_max for plot
        y_max = max(max(arr.values.flatten()) for arr in rmse) * variable_metas[variable]['scale_factor'] * 1.1
        ax.grid()
        ax.legend()
        ax.set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in rmse])} if xlim is None else xlim)
        ax.set_ylim(bottom=0, top=y_max)
        logger.info(f"Saving plot to {plot_file}.")
        fig.savefig(plot_file,dpi=200)

    # return rmse if requested
    if return_rmse:
        return rmse
    else:
        return
    
def acc(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
        return_acc=False,
): 
    """
    Calculates the anomaly correlation coefficeint (ACC) for a set of forecasts and plots and/or returns the results.

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to compare to the forecast. ALso used for calculating the climatology.
        'climatology_file' (str, optional): The file path to the climatology data. If None, the climatology is calculated from the verification data.
            if not None, but file doesn't exist, the climatology is calculated from the verification data and cached to the file.
        'acc_cache' (str, optional): The file path where the calculated ACC should be cached. If None, the ACC is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the ACC is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to None.
    return_acc (bool, optional): If True, the function returns the calculated ACC. Defaults to False.

    Returns:
    list of xarray.DataArray: A list of ACC values for each forecast. Only returned if return_acc is True.
    """

    acc = []
    # iterate through forecats and obtain acc 
    for forecast_param in forecast_params:
        # if acc is cached already, load it
        if os.path.isfile(forecast_param.get('acc_cache','')):
            logger.info(f"Loading ACC from {forecast_param['acc_cache']}.")
            acc.append(xr.open_dataarray(forecast_param['acc_cache']))
        else:
            logger.info(f"Calculating ACC for {forecast_param['file']}.")
        
            # open forecast Yair
            forecast = xr.open_dataset(forecast_param['file'])[variable]
            forecast = forecast.isel(ensemble=0)
            verif = format_verif(xr.open_dataset(forecast_param['verification_file'], engine='zarr'),
                        forecast,
                        variable)

            # calculate climatology
            if forecast_param.get('climatology_file',None) is not None:
                if os.path.isfile(forecast_param['climatology_file']):
                    logger.info(f"Loading climatology from {forecast_param['climatology_file']}")
                    climo_raw = xr.open_dataset(forecast_param['climatology_file'])['targets']
                else:
                    logger.info(f"Calculating climatology from {forecast_param['verification_file']} and caching to {forecast_param['climatology_file']}.")
                    
                    # load verification_data, calculate climo
                    climo_raw = xr.open_dataset(forecast_param['verification_file'],
                                                engine='zarr')['targets'].sel(channel_out=variable).groupby('time.dayofyear').mean(dim='time')
                    # create directory if it doesn't exist
                    os.makedirs(os.path.dirname(forecast_param['climatology_file']), exist_ok=True)
                    climo_raw.to_netcdf(forecast_param['climatology_file'])
            else: 
                logger.info(f"Calculating climatology from {forecast_param['verification_file']}.")
                climo_raw = xr.open_dataset(forecast_param['verification_file'],
                                            engine='zarr')['targets'].sel(channel_out=variable).groupby('time.dayofyear').mean(dim='time')

            # align climo data with forecast data
            logger.info("Aligning and climatology with forecast data.")
            climo = xr.full_like(forecast, fill_value=np.nan)
            for time in forecast.time:
                for step in forecast.step:
                    climo.loc[{'time': time, 'step': step}] = climo_raw.sel(dayofyear=(time + step).dt.dayofyear).values

            # calculate anomalies 
            forec_anom = forecast - climo
            verif_anom = verif - climo
            
            # calculate acc
            axis_mean = ['time', 'face', 'height','width']
            acc.append((verif_anom * forec_anom).mean(dim=axis_mean, skipna=True)
                / np.sqrt((verif_anom**2).mean(dim=axis_mean, skipna=True) *
                          (forec_anom**2).mean(dim=axis_mean, skipna=True)))

            # cache acc if requested
            if forecast_param.get('acc_cache',None) is not None:
                logger.info(f"Caching ACC to {forecast_param['acc_cache']}.")
                # create directory if it doesn't exist
                os.makedirs(os.path.dirname(forecast_param['acc_cache']), exist_ok=True)
                acc[-1].to_netcdf(forecast_param['acc_cache'])

    # plot acc if requested
    if plot_file is not None:
        fig, ax = plt.subplots()
        for skill, plot_kwargs in zip(acc, [forecast_param['plot_kwargs'] for forecast_param in forecast_params]):
            ax.plot([s / np.timedelta64(1, 'D') for s in skill.step.values], # plot in days 
                    skill, # scale to physical units
                    **plot_kwargs) # style curve and label

        # style plot
        ax.set_xlabel('Forecast Days')
        ax.set_ylabel(f'ACC')
        ax.grid()
        ax.legend()
        ax.set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in acc])} if xlim is None else xlim)
        ax.set_ylim(bottom=0, top=1)
        logger.info(f"Saving plot to {plot_file}.")
        fig.savefig(plot_file,dpi=200)

    # return acc if requested
    if return_acc:
        return acc
    else:
        return

def plot_baseline_metrics(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
):
    """
    Uses rmse and acc function to create a two panel baseline metrics plot 

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to compare to the forecast. ALso used for calculating the climatology. Assumes zarr format.
        'climatology_file' (str, optional): The file path to the climatology data. If None, the climatology is calculated from the verification data.
            if not None, but file doesn't exist, the climatology is calculated from the verification data and cached to the file.
        'rmse_cache' (str, optional): The file path where the calculated RMSE should be cached. If None, the RMSE is not cached.
        'acc_cache' (str, optional): The file path where the calculated ACC should be cached. If None, the ACC is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the ACC is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to max leadtime in forecasts.

    Returns:
    None
    """

    # configure plot 
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    
    # get rmses
    rmses = rmse(
        forecast_params=forecast_params,
        variable=variable,
        return_rmse=True,
    )

    # get accs
    accs = acc(
        forecast_params=forecast_params,
        variable=variable,
        return_acc=True,
    )

    # plot rmse and acc, save 
    logger.info(f"Plotting metrics and saving to {plot_file}.")
    for i in range(len(forecast_params)):
        # rmse
        axs[0].plot([s / np.timedelta64(1, 'D') for s in rmses[i].step.values], # plot in days 
                    rmses[i] * variable_metas[variable]['scale_factor'], # scale to physical units
                    # rmses[i][0,:] * variable_metas[variable]['scale_factor'], # scale to physical units
                    **forecast_params[i]['plot_kwargs']) # convey kwargs
        # acc
        axs[1].plot([s / np.timedelta64(1, 'D') for s in accs[i].step.values], # plot in days 
                    # accs[i][0,:],
                    accs[i],
                    **forecast_params[i]['plot_kwargs'])

    # style rmse plot 
    axs[0].set_xlabel('Forecast Days')
    axs[0].set_ylabel(f'RMSE [{variable_metas[variable]["units"]}]')
    y_max = max(max(arr.values.flatten()) for arr in rmses) * variable_metas[variable]['scale_factor'] * 1.1 # calculate y_max for plot
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in rmses])} if xlim is None else xlim)
    axs[0].set_ylim(bottom=0, top=y_max)

    # style acc plot
    axs[1].set_xlabel('Forecast Days')
    axs[1].set_ylabel(f'ACC')
    axs[1].grid()
    axs[1].legend()
    axs[1].set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in accs])} if xlim is None else xlim)
    axs[1].set_ylim(bottom=0, top=1.05)

    fig.savefig(plot_file,dpi=200)

# example call to plot_baseline_metrics
if __name__ == '__main__':

    plot_baseline_metrics(
        forecast_params=[
            {
                'file':"/pscratch/sd/y/yacohen/nvdlesm/inference_1014/forecast_atmos.nc",
                'verification_file':'/global/cfs/projectdirs/m4331/datasets/era5_hpx/era5_hpx32_8var-coupled_6h_24h.zarr',
                'climatology_file':'./climatology_cache.nc',
                'rmse_cache':'./rmse_cache.nc',
                'acc_cache':'./acc_cache.nc',
                'plot_kwargs':{'label':'model_name','color':'blue','linewidth':2, 'linestyle':'--'},  
            }
        ],
        variable='t850',
        plot_file="./score_rmse_acc_hpx32_det_t850.png",
    )

