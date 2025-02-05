import logging
import argparse
import os
from pathlib import Path
import time
import json
from types import SimpleNamespace
from hydra import initialize, compose
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import dask.array
import numpy as np
import pandas as pd
import torch as th
import xarray as xr
from tqdm import tqdm
import pprint
from torchinfo import summary
from dask.diagnostics import ProgressBar
import itertools
from modulus import Module
from ensemble_utils import bred_vector
import earth2grid
from scipy import interpolate
from multiprocessing import Pool, Manager, cpu_count
import multiprocessing

# Set the start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

def configure_logging(verbose=1):
    verbose_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
            3: logging.NOTSET
            }
    if verbose not in verbose_levels.keys():
        verbose = 1
        current_logger = logging.getLogger()
        current_logger.setLevel(verbose_levels[verbose])
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s][PID=%(process)d]"
            "[%(levelname)s %(filename)s:%(lineno)d] - %(message)s"))
        handler.setLevel(verbose_levels[verbose])
        current_logger.addHandler(handler)

def to_chunked_dataset(ds, chunking):
    """
    Create a chunked copy of a Dataset with proper encoding for netCDF export.
    :param ds: xarray.Dataset
    :param chunking: dict: chunking dictionary as passed to
    xarray.Dataset.chunk()
    :return: xarray.Dataset: chunked copy of ds with proper encoding
    """
    chunk_dict = dict(ds.dims)
    chunk_dict.update(chunking)
    for var in ds.data_vars:
        if 'coordinates' in ds[var].encoding:
            del ds[var].encoding['coordinates']
        ds[var].encoding['contiguous'] = False
        ds[var].encoding['original_shape'] = ds[var].shape
        ds[var].encoding['chunksizes'] = tuple([chunk_dict[d] for d in ds[var].dims])
        ds[var].encoding['chunks'] = tuple([chunk_dict[d] for d in ds[var].dims])
    return ds

def encode_variables_as_int(ds, dtype='int16', compress=0, exclude_vars=()):
    """
    Adds encoding to Dataset variables to export to int16 type instead of float at write time.
    :param ds: xarray Dataset
    :param dtype: as understood by numpy
    :param compress: int: if >1, also enable compression of the variable with this compression level
    :param exclude_vars: iterable of variable names which are not to be encoded
    :return: xarray Dataset with encoding applied
    """
    for var in ds.data_vars:
        if var in exclude_vars:
            continue
        var_max = float(ds[var].max())
        var_min = float(ds[var].min())
        var_offset = (var_max + var_min) / 2
        var_scale = (var_max - var_min) / (2 * np.iinfo(dtype).max)
        if var_scale == 0:
            logger.warning("min and max for variable %s are both %f", var, var_max)
            var_scale = 1.
        ds[var].encoding.update({
            'scale_factor': var_scale,
            'add_offset': var_offset,
            'dtype': dtype,
            '_Unsigned': not np.issubdtype(dtype, np.signedinteger),
            '_FillValue': np.iinfo(dtype).min,
        })
        if 'valid_range' in ds[var].attrs:
            del ds[var].attrs['valid_range']
        if compress > 0:
            ds[var].encoding.update({
                'zlib': True,
                'complevel': compress
            })
    return ds

def _convert_time_step(dt):  # pylint: disable=invalid-name
    return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)

def get_forecast_dates(start, end, freq):
    # Get the dates parameters
    if freq == 'biweekly':
        dates_1 = pd.date_range(start, end, freq='7D')
        dates_2 = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=3), end, freq='7D')
        dates = dates_1.append(dates_2).sort_values().to_numpy()
    else:
        dates = pd.date_range(start, end, freq=freq).to_numpy()
    return dates

def read_forecast_dates_from_file(path):
    import glob
    file_paths = glob.glob(os.path.join(path, "*.txt"))
    all_dates = []
    for file_path in file_paths:
        with open(file_path) as file:
            dates = file.read().splitlines()
            for date in dates:
                if not np.datetime64("2000-01-01") < np.datetime64(date) < np.datetime64("2021-01-01"): continue
                # Create numpy datetime64 at t-8, t-7, t-6, t-5, t-4 days
                all_dates += list(pd.date_range(start=pd.Timestamp(date)-pd.Timedelta(8, "D"),
                    end=pd.Timestamp(date)-pd.Timedelta(4, "D"),
                    freq=pd.Timedelta(1, "D")).to_numpy())
    return np.unique(np.sort(np.array(all_dates)))


def get_latest_version(directory):
    all_versions = [os.path.join(directory, v) for v in os.listdir(directory)]
    all_versions = [v for v in all_versions if os.path.isdir(v)]
    latest_version = max(all_versions, key=os.path.getmtime)
    return Path(latest_version).name


def get_coupled_time_dim(atmos_cfg, ocean_cfg):
    
    atmos_step = pd.Timedelta(atmos_cfg.data.gap)+(atmos_cfg.data.input_time_dim-1)*pd.Timedelta(atmos_cfg.data.time_step)
    ocean_step = pd.Timedelta(ocean_cfg.data.gap)+(ocean_cfg.data.input_time_dim-1)*pd.Timedelta(ocean_cfg.data.time_step)
    intersection = pd.Timedelta(np.lcm(atmos_step.value, ocean_step.value),units='ns')
    
    ocean_output_time_dim = (intersection // ocean_step)*ocean_cfg.data.input_time_dim
    atmos_output_time_dim = (intersection // atmos_step)*atmos_cfg.data.input_time_dim
    
    return atmos_output_time_dim, ocean_output_time_dim


def interpolate_sst(ocean_output, lsm):
    if isinstance(lsm, th.Tensor):
        lsm = lsm.cpu().numpy()  # Ensure it's on CPU before converting

    # Extract relevant dimensions
    steps = ocean_output.size(2)  
    ensembles = ocean_output.size(0)
    width = ocean_output.size(-1)
    level = int(np.log2(width))
    
    # Create HEALPIX grid
    hpx = earth2grid.healpix.Grid(
        level=level,
        pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
    )
    
    # Determine the number of latitudes and longitudes based on HEALPIX grid resolution
    nlat = 2 ** level  # Latitudes based on HEALPIX level
    nlon = 2 * nlat    # Longitudes (HEALPIX grid covers the globe, so 4x the lat resolution is reasonable)

    # Create regular lat-lon grid
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(-180, 180, nlon, endpoint=False)
    ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    # Loop over ensemble members and steps to apply SST interpolation
    for ensemble in range(ensembles):
        for step in range(steps):
            # Extract the SST data with dimensions (face=12, height=32, width=32)
            sst = ocean_output[ensemble, :, step, 0, :, :].cpu().numpy()  # Extracting SST tensor (for all faces)
            
            # Regrid SST and LSM to regular lat-lon grid
            data_flat_sst = th.from_numpy(sst.flatten()).to(th.float64)
            data_flat_lsm = th.from_numpy(lsm.flatten()).to(th.float64)
            
            regrid = earth2grid.get_regridder(hpx, ll_grid)
            sst_regridded = regrid(data_flat_sst).reshape(nlat, nlon).numpy()
            lsm_regridded = regrid(data_flat_lsm).reshape(nlat, nlon).numpy()
            
            # Initialize sst_interp with NaNs
            sst_interp = np.full_like(sst_regridded, np.nan)
            
            # Perform zonal interpolation
            for i in range(nlat):
                sst_row = sst_regridded[i, :]
                lsm_row = lsm_regridded[i, :]
                lon_row = lons  # from -180 to 180 degrees
                
                # Find ocean points
                ocean_mask = (lsm_row == 0) & (~np.isnan(sst_row))
                ocean_lons = lon_row[ocean_mask]
                ocean_sst = sst_row[ocean_mask]
                
                if len(ocean_sst) > 1:
                    # Wrap the data around the -180/180 discontinuity
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
                lat_indices_to_fill = np.arange(southernmost_valid_lat_index + 1, nlat)
                
                if len(lat_indices_to_fill) > 0:
                    for i in lat_indices_to_fill:
                        sst_interp[i, :] = southernmost_valid_sst
                else:
                    print("No latitudes to fill south of the southernmost valid latitude.")
            else:
                print("Warning: No valid SST data found to fill NaN values south of the southernmost valid latitude.")
            
            # Regrid back to HEALPIX grid
            data_interp_tensor = th.from_numpy(sst_interp).to(th.float64)
            regrid_back = earth2grid.get_regridder(ll_grid, hpx)
            sst_interp_hpx_flat = regrid_back(data_interp_tensor)
            sst_interp_hpx_ = sst_interp_hpx_flat.reshape(12, 2**level, 2**level).numpy()
            
            # Verify that values over the ocean are unchanged
            ocean_mask = ~np.isnan(sst)
            sst_interp_hpx = np.copy(sst_interp_hpx_)
            sst_interp_hpx[ocean_mask] = sst[ocean_mask]
            
            # Replace the SST in ocean_output with the interpolated SST
            ocean_output[ensemble, :, step, 0, :, :] = th.from_numpy(sst_interp_hpx).to(ocean_output.device)
    return ocean_output


def inference_checkpoint(atmos_model, ocean_model, atmos_data_module, ocean_data_module, 
                         forecast_dates, forecast_integrations, atmos_output_lead_times, ocean_output_lead_times, args):
    device = next(atmos_model.parameters()).device
    atmos_coupled_time_dim = atmos_model.output_time_dim
    ocean_coupled_time_dim = ocean_model.output_time_dim

    atmos_coupler = atmos_data_module.test_dataset.couplings[0]
    atmos_coupler.setup_coupling(ocean_data_module)
    ocean_coupler = ocean_data_module.test_dataset.couplings[0]
    ocean_coupler.setup_coupling(atmos_data_module)

    atmos_loader_iter = iter(atmos_data_module.test_dataloader()[0])
    ocean_loader_iter = iter(ocean_data_module.test_dataloader()[0])

    noise_dim = args.noise_num
    noise_var = args.noise_var
    
    atmos_prediction_noise = np.empty((noise_dim, len(forecast_dates),
        len(atmos_output_lead_times) + 1,
        len(atmos_data_module.output_variables)) + atmos_data_module.test_dataset.spatial_dims,
        dtype='float32')
    ocean_prediction_noise = np.empty((noise_dim, len(forecast_dates),
        len(ocean_output_lead_times) + 1,
        len(ocean_data_module.output_variables)) + ocean_data_module.test_dataset.spatial_dims,
        dtype='float32')
    
    atmos_constants = None
    ocean_constants = None

    pbar = tqdm(forecast_dates)
    for i, init in enumerate(pbar):
        pbar.postfix = pd.Timestamp(forecast_dates[i]).strftime('init %Y-%m-%d %HZ')
        pbar.update()
        new_init = True
        atmos_coupler.reset_coupler()
        ocean_coupler.reset_coupler()

        for j in range(forecast_integrations):
            if j == 0:
                atmos_input = [k.to(device) for k in next(atmos_loader_iter)]
                atmos_input[0] = atmos_input[0].expand(noise_dim+1, *atmos_input[0].shape[1:]).clone()
                    # atmos_input[0]= atmos channels: dims `batch`, `face`,  step, channel, `height`, `width`
                atmos_input[1] = atmos_input[1].expand(noise_dim+1, *atmos_input[1].shape[1:]).clone()
                    # atmos_input[1] = positional embbedings: dims `batch`, `face`,  `positional-embbedings`, `channel`, `height`, `width`
                    # positional-embbedings = channel*2 + leadtime ?
                atmos_input[3] = atmos_input[3].expand(atmos_input[3].shape[0], noise_dim+1, *atmos_input[3].shape[2:]).clone()
                    # atmos_input[3] = coupling channels (sst): dims `coupled integration dim`, `batch`, `channel`, `face`, `height`, `width`

                if atmos_constants is None:
                    atmos_constants = atmos_input[2]
                with th.no_grad():
                    atmos_output = atmos_model(atmos_input)
                
                ocean_coupler.set_coupled_fields(atmos_output.cpu())
                ocean_input = [k.to(device) for k in next(ocean_loader_iter)]
                ocean_input[0] = ocean_input[0].expand(noise_dim+1, *ocean_input[0].shape[1:]).clone()
                ocean_input[1] = ocean_input[1].expand(noise_dim+1, *ocean_input[1].shape[1:]).clone()

                if args.perturbation_method == 'gaussian':
                    for k in range(noise_dim):
                        ocean_input[0][k+1] += np.sqrt(noise_var)*th.randn(*ocean_input[0].shape[1:]).to(device)
                elif args.perturbation_method == 'bred_vector':
                    ocean_input[0] = bred_vector(
                        ocean_input, ocean_model, atmos_model, atmos_input, atmos_coupler, ocean_coupler,
                        atmos_data_module, ocean_data_module, ocean_loader_iter,
                        atmos_constants, ocean_constants,
                        noise_dim, noise_var, device, integration_steps=args.bred_vector_steps, inflate=False
                    )
                        
                if ocean_constants is None:
                    ocean_constants = ocean_input[2]
                with th.no_grad():
                    ocean_output = ocean_model(ocean_input)
            else:
                atmos_coupler.set_coupled_fields(ocean_output.cpu())
                atmos_input = [k.to(device) for k in atmos_data_module.test_dataset.next_integration(
                                                        atmos_output, 
                                                        constants = atmos_constants,
                                                    )]
                atmos_input[1] = atmos_input[1].expand(noise_dim+1, *atmos_input[1].shape[1:]).clone()

                with th.no_grad():
                    atmos_output = atmos_model(atmos_input)

                ocean_coupler.set_coupled_fields(atmos_output.cpu())
                ocean_input = [k.to(device) for k in ocean_data_module.test_dataset.next_integration(
                                                        ocean_output, 
                                                        constants = ocean_constants,
                                                    )]
                ocean_input[1] = ocean_input[1].expand(noise_dim+1, *ocean_input[1].shape[1:]).clone()

                with th.no_grad():
                    ocean_output = ocean_model(ocean_input)
                    ocean_output = interpolate_sst(ocean_output, ocean_constants)

            if new_init:
                for k in range(noise_dim):
                    ocean_prediction_noise[k, i:i+1][:, 0] = ocean_input[0][k+1:k+2].permute(0, 2, 3, 1, 4, 5)[:, -1].cpu().numpy()
                    atmos_prediction_noise[k, i:i+1][:, 0] = atmos_input[0][k+1:k+2].permute(0, 2, 3, 1, 4, 5)[:, -1].cpu().numpy()
            atmos_input[0] = atmos_input[0][0:1]
            ocean_input[0] = ocean_input[0][0:1]
            atmos_output_noise = atmos_output.clone()
            atmos_output = atmos_output[0:1].clone()
            ocean_output_noise = ocean_output.clone()
            ocean_output = ocean_output[0:1].clone()
            ocean_prediction_noise[:, i:i+1][:,:,slice(1+j*ocean_coupled_time_dim,(j+1)*ocean_coupled_time_dim+1)] = ocean_output_noise[1:].permute(0, 2, 3, 1, 4, 5).unsqueeze(1).cpu().numpy()
            atmos_prediction_noise[:, i:i+1][:,:,slice(1+j*atmos_coupled_time_dim,(j+1)*atmos_coupled_time_dim+1)] = atmos_output_noise[1:].permute(0, 2, 3, 1, 4, 5).unsqueeze(1).cpu().numpy()
            ocean_output = ocean_output_noise
            atmos_output = atmos_output_noise

    return atmos_prediction_noise, ocean_prediction_noise



def worker(args):
    ckpt_idx, atmos_ckpt, ocean_ckpt, gpu_id, atmos_coupled_time_dim, ocean_coupled_time_dim, atmos_data_module, ocean_data_module, forecast_dates, forecast_integrations, atmos_output_lead_times, ocean_output_lead_times, args = args
    try:
        print(f"Worker started: ckpt_idx={ckpt_idx}, gpu_id={gpu_id}")
        
        # Initialize CUDA for this process
        th.cuda.init()
        th.cuda.set_device(gpu_id)
        device = th.device(f'cuda:{gpu_id}')

        # Load models for this checkpoint
        atmos_model = Module.from_checkpoint(str(atmos_ckpt))
        atmos_model.output_time_dim = atmos_coupled_time_dim
        atmos_model = atmos_model.to(device)
        atmos_model.eval()

        ocean_model = Module.from_checkpoint(str(ocean_ckpt))
        ocean_model.output_time_dim = ocean_coupled_time_dim
        ocean_model = ocean_model.to(device)
        ocean_model.eval()

        # Run inference for this checkpoint combination
        atmos_prediction_noise, ocean_prediction_noise = inference_checkpoint(
            atmos_model, ocean_model, atmos_data_module, ocean_data_module,
            forecast_dates, forecast_integrations, atmos_output_lead_times, ocean_output_lead_times, args
        )

        print(f"Worker completed: ckpt_idx={ckpt_idx}, gpu_id={gpu_id}")
        return ckpt_idx, atmos_prediction_noise, ocean_prediction_noise
    except Exception as e:
        print(f"Error in worker (ckpt_idx={ckpt_idx}, gpu_id={gpu_id}): {str(e)}")
        import traceback
        print(traceback.format_exc())
        return ckpt_idx, None, None


def coupled_inference(args: argparse.Namespace):
    forecast_dates = get_forecast_dates(args.forecast_init_start, args.forecast_init_end, args.freq)
    os.makedirs(args.output_directory, exist_ok=True)

    if args.num_gpus == -1:
        device = th.device('cpu')
    else:
        device = th.device(f'cuda:{args.num_gpus}' if th.cuda.is_available() else 'cpu')
    with initialize(config_path=os.path.join(args.atmos_hydra_path, '.hydra'), version_base=None):
        atmos_cfg = compose('config.yaml')
    with initialize(config_path=os.path.join(args.ocean_hydra_path, '.hydra'), version_base=None):
        ocean_cfg = compose('config.yaml')

    batch_size = 1
    # override config params for forecasting 
    atmos_cfg.num_workers = 0
    atmos_cfg.batch_size = batch_size
    atmos_cfg.data.prebuilt_dataset = True
    ocean_cfg.num_workers = 0
    ocean_cfg.batch_size = batch_size
    ocean_cfg.data.prebuilt_dataset = True

    # some models do not have custom cuda healpix padding flags in config, instead they assume default behavior of model class
    # here we ensure that this default behaviopr is overridden fore forecasting 
    if not hasattr(atmos_cfg.model,'enable_healpixpad'):
        OmegaConf.set_struct(atmos_cfg,True)
        with open_dict(atmos_cfg):
            atmos_cfg.model.enable_healpixpad = False
    else:
        atmos_cfg.model.enable_healpixpad = False
    if not hasattr(ocean_cfg.model,'enable_healpixpad'):
        OmegaConf.set_struct(ocean_cfg,True)
        with open_dict(ocean_cfg):
            ocean_cfg.model.enable_healpixpad = False
    else:
        ocean_cfg.model.enable_healpixpad = False
 
    # Set up data module with some overrides for inference. Compute expected output time dimension.
    atmos_output_lead_times = np.arange(
        _convert_time_step(atmos_cfg.data.gap),
        _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
        _convert_time_step(atmos_cfg.data.time_step)
    )
    ocean_output_lead_times = np.arange(
        _convert_time_step(ocean_cfg.data.gap),
        _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
        _convert_time_step(ocean_cfg.data.time_step)
    )
    # figure concurrent forecasting time variables 
    atmos_coupled_time_dim, ocean_coupled_time_dim = get_coupled_time_dim(atmos_cfg, ocean_cfg)
    # The number of times each model will be called. Should be the same whether ocean or atmos 
    # is used to make calculation 
    forecast_integrations = len(ocean_output_lead_times) // ocean_coupled_time_dim
    # set up couplers for forecasting 
    try:
        nc = len(atmos_cfg.data.module.couplings)
        for i in range(nc):
            atmos_cfg.data.module.couplings[i]['params']['output_time_dim'] = atmos_coupled_time_dim 
    except AttributeError:
        print(f'model {args.atmos_model_path} is not interpreted as a coupled model, cannot perform coupled forecast. Aborting.')
    try:
        nc = len(ocean_cfg.data.module.couplings)
        for i in range(nc):
            ocean_cfg.data.module.couplings[i]['params']['output_time_dim'] = ocean_coupled_time_dim 
    except AttributeError:
        print(f'model {args.atmos_model_path} is not interpreted as a coupled model, cannot perform coupled forecast. Aborting.')
    
    optional_kwargs = {k: v for k, v in {
        'dst_directory': args.data_directory,
        'prefix': args.data_prefix,
        'suffix': args.data_suffix
    }.items() if v is not None}
    
    # instantiate data modules 
    atmos_data_module = instantiate(
        atmos_cfg.data.module,
        output_time_dim=atmos_coupled_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
        **optional_kwargs
    )
    atmos_loader, _ = atmos_data_module.test_dataloader()
    ocean_data_module = instantiate(
        ocean_cfg.data.module,
        output_time_dim=ocean_coupled_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
        **optional_kwargs
    )
    ocean_loader, _ = ocean_data_module.test_dataloader()
    
    # checks to make sure timeing and lead time line up 
    if forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step) != \
        forecast_integrations*atmos_coupled_time_dim*pd.Timedelta(atmos_data_module.time_step):
        raise ValueError('Lead times of atmos and ocean models does not align.')
    if forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step) != _convert_time_step(args.lead_time):
        raise ValueError(f'Requested leadtime ({_convert_time_step(args.lead_time)}) and coupled integration ({forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step)}) are not the same. Make sure lead time is compatible with component model intersections.')


    # Set output_time_dim param override.
    atmos_input_channels = len(atmos_cfg.data.input_variables)
    atmos_output_channels = len(atmos_cfg.data.output_variables) if atmos_cfg.data.output_variables is not None else atmos_input_channels
    atmos_constants_arr = atmos_data_module.constants
    atmos_n_constants = 0 if atmos_constants_arr is None else len(atmos_constants_arr.keys()) # previously was 0 but with new format it is 1
    ocean_input_channels = len(ocean_cfg.data.input_variables)
    ocean_output_channels = len(ocean_cfg.data.output_variables) if ocean_cfg.data.output_variables is not None else ocean_input_channels
    ocean_constants_arr = ocean_data_module.constants
    ocean_n_constants = 0 if ocean_constants_arr is None else len(ocean_constants_arr.keys()) # previously was 0 but with new format it is 1

    atmos_decoder_input_channels = int(atmos_cfg.data.get('add_insolation', 0))
    atmos_cfg.model['input_channels'] = atmos_input_channels
    atmos_cfg.model['output_channels'] = atmos_output_channels
    atmos_cfg.model['n_constants'] = atmos_n_constants
    atmos_cfg.model['decoder_input_channels'] = atmos_decoder_input_channels
    ocean_decoder_input_channels = int(ocean_cfg.data.get('add_insolation', 0))
    ocean_cfg.model['input_channels'] = ocean_input_channels
    ocean_cfg.model['output_channels'] = ocean_output_channels
    ocean_cfg.model['n_constants'] = ocean_n_constants
    ocean_cfg.model['decoder_input_channels'] = ocean_decoder_input_channels
    
    # instantiate models and find best checkpoint 
    atmos_model_name = Path(args.atmos_model_path).name
    atmos_checkpoint_basepath = os.path.join(args.atmos_model_path, "tensorboard", "checkpoints")
    if args.atmos_model_checkpoints is None:
        raise ValueError('atmosphere model checkpoint name must be given')
    else:
        atmos_checkpoint_path = os.path.join(atmos_checkpoint_basepath, args.atmos_model_checkpoints[0])
    logger.info("load model checkpoint %s", atmos_checkpoint_path)
    ocean_model_name = Path(args.ocean_model_path).name
    ocean_checkpoint_basepath = os.path.join(args.ocean_model_path, "tensorboard", "checkpoints")
    if args.ocean_model_checkpoints is None:
        raise ValueError('ocean model checkpoint name must be given')
    else:
        ocean_checkpoint_path = os.path.join(ocean_checkpoint_basepath, args.ocean_model_checkpoints[0])
    logger.info("load model checkpoint %s", ocean_checkpoint_path)

    # Load checkpoints
    atmos_checkpoints = [os.path.join(args.atmos_model_path, "tensorboard", "checkpoints", ckpt) for ckpt in args.atmos_model_checkpoints]
    ocean_checkpoints = [os.path.join(args.ocean_model_path, "tensorboard", "checkpoints", ckpt) for ckpt in args.ocean_model_checkpoints]

    # Generate all combinations of atmosphere and ocean checkpoints
    checkpoint_combinations = list(itertools.product(atmos_checkpoints, ocean_checkpoints))

    # Calculate the total number of combinations
    # num_combinations = len(checkpoint_combinations)
    num_gpus = min(args.num_gpus, th.cuda.device_count())

    if num_gpus > 1:
        # Use multiple GPUs
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=num_gpus)

        # Prepare arguments for pool.map
        args_list = [
            (ckpt_idx, atmos_ckpt, ocean_ckpt, ckpt_idx % num_gpus, 
             atmos_coupled_time_dim, ocean_coupled_time_dim, 
             atmos_data_module, ocean_data_module, 
             forecast_dates, forecast_integrations, 
             atmos_output_lead_times, ocean_output_lead_times, args)
            for ckpt_idx, (atmos_ckpt, ocean_ckpt) in enumerate(checkpoint_combinations)
        ]

        # Use pool.map to distribute work across GPUs
        results = pool.map(worker, args_list)

        # Close the pool
        pool.close()
        pool.join()

        # Process results
        atmos_ckpt_predictions = []
        ocean_ckpt_predictions = []
        ckpt_order = []

        for ckpt_idx, atmos_pred, ocean_pred in results:
            if atmos_pred is not None and ocean_pred is not None:
                atmos_ckpt_predictions.append(atmos_pred)
                ocean_ckpt_predictions.append(ocean_pred)
                ckpt_order.append(ckpt_idx)

        # Reorder predictions based on the original checkpoint order
        sorted_indices = sorted(range(len(ckpt_order)), key=lambda k: ckpt_order[k])
        atmos_ckpt_predictions = [atmos_ckpt_predictions[i] for i in sorted_indices]
        ocean_ckpt_predictions = [ocean_ckpt_predictions[i] for i in sorted_indices]

        print(f"Processed {len(atmos_ckpt_predictions)} out of {len(checkpoint_combinations)} checkpoints")

    else:
        # Use a single GPU or CPU
        device = th.device(f'cuda:0' if th.cuda.is_available() and num_gpus > 0 else 'cpu')
        
        atmos_ckpt_predictions = []
        ocean_ckpt_predictions = []
        for ckpt_idx, (atmos_ckpt, ocean_ckpt) in enumerate(checkpoint_combinations):
            # Load models for this checkpoint
            atmos_model = Module.from_checkpoint(str(atmos_ckpt))
            atmos_model.output_time_dim = atmos_coupled_time_dim
            atmos_model = atmos_model.to(device)
            atmos_model.eval()

            ocean_model = Module.from_checkpoint(str(ocean_ckpt))
            ocean_model.output_time_dim = ocean_coupled_time_dim
            ocean_model = ocean_model.to(device)
            ocean_model.eval()

            # Run inference for this checkpoint combination
            atmos_prediction_noise, ocean_prediction_noise = inference_checkpoint(
                atmos_model, ocean_model, atmos_data_module, ocean_data_module,
                forecast_dates, forecast_integrations, atmos_output_lead_times, ocean_output_lead_times, args
            )
            
            # Store predictions for this checkpoint
            atmos_ckpt_predictions.append(atmos_prediction_noise)
            ocean_ckpt_predictions.append(ocean_prediction_noise)

    # Combine predictions from all checkpoints
    atmos_prediction_noise = np.stack(atmos_ckpt_predictions)
    ocean_prediction_noise = np.stack(ocean_ckpt_predictions)

    # Generate dataarray with coordinates
    ocean_meta_ds = ocean_data_module.test_dataset.ds
    atmos_meta_ds = atmos_data_module.test_dataset.ds

    dims = ['ckpt_ensemble', 'ic_ensemble', 'time', 'step', 'channel_out', 'face', 'height', 'width']

    atmos_coords = {
        'ckpt_ensemble': range(len(checkpoint_combinations)),
        'ic_ensemble': range(atmos_prediction_noise.shape[1]),
        'time': forecast_dates,
        'step': [pd.Timedelta(hours=0)] + list(atmos_output_lead_times),
        'channel_out': atmos_cfg.data.output_variables or atmos_cfg.data.input_variables,
        'face': atmos_meta_ds.face,
        'height': atmos_meta_ds.height,
        'width': atmos_meta_ds.width
    }

    ocean_coords = {
        'ckpt_ensemble': range(len(checkpoint_combinations)),
        'ic_ensemble': range(ocean_prediction_noise.shape[1]),
        'time': forecast_dates,
        'step': [pd.Timedelta(hours=0)] + list(ocean_output_lead_times),
        'channel_out': ocean_cfg.data.output_variables or ocean_cfg.data.input_variables,
        'face': ocean_meta_ds.face,
        'height': ocean_meta_ds.height,
        'width': ocean_meta_ds.width
    }


    ocean_prediction_da = xr.DataArray(ocean_prediction_noise, dims=dims, coords=ocean_coords)
    atmos_prediction_da = xr.DataArray(atmos_prediction_noise, dims=dims, coords=atmos_coords)
    # Re-scale prediction
    ocean_prediction_da[:] *= ocean_data_module.test_dataset.target_scaling['std']
    ocean_prediction_da[:] += ocean_data_module.test_dataset.target_scaling['mean']
    ocean_prediction_ds = ocean_prediction_da.to_dataset(dim='channel_out')
    for variable in ocean_prediction_ds.data_vars:
        if ocean_cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            ocean_prediction_ds[variable] = np.exp(
                ocean_prediction_ds[variable] + np.log(ocean_cfg.data.scaling[variable]['log_epsilon'])
            ) - ocean_cfg.data.scaling[variable]['log_epsilon']
    atmos_prediction_da[:] *= atmos_data_module.test_dataset.target_scaling['std']
    atmos_prediction_da[:] += atmos_data_module.test_dataset.target_scaling['mean']
    atmos_prediction_ds = atmos_prediction_da.to_dataset(dim='channel_out')
    for variable in atmos_prediction_ds.data_vars:
        if atmos_cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            atmos_prediction_ds[variable] = np.exp(
                atmos_prediction_ds[variable] + np.log(atmos_cfg.data.scaling[variable]['log_epsilon'])
            ) - atmos_cfg.data.scaling[variable]['log_epsilon']

    # Add args as metadata to the datasets
    args_dict = vars(args)
    ocean_prediction_ds.attrs['config'] = str(args_dict)
    atmos_prediction_ds.attrs['config'] = str(args_dict)

    args_dict.pop('ocean_model_checkpoints', None)
    args_dict.pop('atmos_model_checkpoints', None)
    args_dict['checkpoint_combinations'] = [
        {
            'ocean_checkpoint': ocean_ckpt,
            'atmos_checkpoint': atmos_ckpt
        }
        for ocean_ckpt, atmos_ckpt in checkpoint_combinations
    ]
    
    # Export dataset
    write_time = time.time()
    ocean_prediction_ds = to_chunked_dataset(ocean_prediction_ds, {'time': 8})
    atmos_prediction_ds = to_chunked_dataset(atmos_prediction_ds, {'time': 8})
    if args.encode_int:
        ocean_prediction_ds = encode_variables_as_int(ocean_prediction_ds, compress=1)
        atmos_prediction_ds = encode_variables_as_int(atmos_prediction_ds, compress=1)

    if getattr(args,'ocean_output_filename',None) is not None:
        ocean_output_file = os.path.join(args.output_directory, f"{args.ocean_output_filename}.{'zarr' if args.to_zarr else 'nc'}")
    if getattr(args,'atmos_output_filename',None) is not None:
        atmos_output_file = os.path.join(args.output_directory, f"{args.atmos_output_filename}.{'zarr' if args.to_zarr else 'nc'}")
    else:
        ocean_output_file = os.path.join(args.output_directory, f"forecast_{ocean_model_name}.{'zarr' if args.to_zarr else 'nc'}")
        atmos_output_file = os.path.join(args.output_directory, f"forecast_{atmos_model_name}.{'zarr' if args.to_zarr else 'nc'}")
    logger.info(f"writing forecasts to {atmos_output_file} and {ocean_output_file}")
    if args.to_zarr:
        ocean_write_job = ocean_prediction_ds.to_zarr(ocean_output_file, compute=False)
        atmos_write_job = atmos_prediction_ds.to_zarr(atmos_output_file, compute=False)
    else:
        ocean_write_job = ocean_prediction_ds.to_netcdf(ocean_output_file, compute=False)
        atmos_write_job = atmos_prediction_ds.to_netcdf(atmos_output_file, compute=False)
    with ProgressBar():
        ocean_write_job.compute()
        atmos_write_job.compute()
    logger.debug("wrote file in %0.1f s", time.time() - write_time)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce forecasts from a DLWP model.')
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the JSON configuration file")
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    config = load_config(args.config)
    
    # Convert config dictionary to Namespace object
    run_args = SimpleNamespace(**config)
    run_args.atmos_subset_channels = config.get('atmos_subset_channels', None)
    run_args.ocean_subset_channels = config.get('ocean_subset_channels', None)
    
    # Configure logging
    configure_logging(2)
    
    # Process paths for Hydra
    run_args.atmos_hydra_path = os.path.relpath(run_args.atmos_model_path, os.path.join(os.getcwd(), ''))
    run_args.ocean_hydra_path = os.path.relpath(run_args.ocean_model_path, os.path.join(os.getcwd(), ''))

    # Process checkpoint paths
    run_args.atmos_model_checkpoints = [
        os.path.join(run_args.atmos_model_path, ckpt) for ckpt in run_args.atmos_model_checkpoints
    ]
    run_args.ocean_model_checkpoints = [
        os.path.join(run_args.ocean_model_path, ckpt) for ckpt in run_args.ocean_model_checkpoints
    ]

    logger.debug("model paths: %s", (run_args.atmos_model_path, run_args.ocean_model_path))
    logger.debug("python working dir: %s", os.getcwd())
    logger.debug("hydra paths: %s", (run_args.atmos_hydra_path, run_args.ocean_hydra_path))
    logger.debug("atmos checkpoints: %s", run_args.atmos_model_checkpoints)
    logger.debug("ocean checkpoints: %s", run_args.ocean_model_checkpoints)

    # Call the coupled_inference function with run_args
    coupled_inference(run_args)

