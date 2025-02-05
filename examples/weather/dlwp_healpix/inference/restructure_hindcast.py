import os
import gc
import xarray as xr
import numpy as np

def restructure_hindcast(
    source_dir,
    output_dir,
    years=range(1996, 2016),
    variable="t2m",
    reference_year=1996,
    chunks=None
):
    """
    For each month-day pair found in the reference year's file, gather
    that same month-day from all 'years' and produce one file with
    dimension 'time' stacking the selected variable data across years.

    The result for each month-day will be written to:
        {output_dir}/{variable}/hindcast_{mm}-{dd}.nc

    Assumptions/Modifications:
    1) We assume there is exactly one time slot per day/month in each file.
       The code will raise an error if more than one time slot is found.
    2) We rename the 'ckpt_ensemble' dimension to 'number', preserving
       the ensemble dimension consistent with the ECMWF naming.
    3) We explicitly close files/datasets and call gc.collect() after each
       iteration to help free memory.

    Parameters
    ----------
    source_dir : str
        Path to the folder that contains hindcast_<year>/forecast_atmos.nc
    output_dir : str
        Output folder to store reorganized NetCDFs
    years : iterable
        Sequence of years to process (e.g. range(1996, 2016))
    variable : str
        The variable to extract (e.g. "t2m", "t850", or "z500")
    reference_year : int
        We use the file from this year to discover valid month-day pairs
        present in the 'time' dimension.
    chunks : dict or None
        If provided, xarray will open files as dask arrays with the specified
        chunking, e.g. chunks={'time': 1, 'face': 12, 'height':64, 'width':64}.
        This can help avoid loading huge data into memory.
    """

    import dask

    # Optionally, you can tweak dask settings (e.g., to use fewer threads) if desired:
    # dask.config.set(scheduler='threads', num_workers=2)

    # Make sure the subfolder for this variable exists
    var_output_path = os.path.join(output_dir, variable)
    os.makedirs(var_output_path, exist_ok=True)

    # ---------------------------------------------------------
    # 1) Get the reference dataset for the reference year
    # ---------------------------------------------------------
    ref_file = os.path.join(source_dir, f"hindcast_{reference_year}", "forecast_atmos.nc")

    if chunks is not None:
        ds_ref = xr.open_dataset(ref_file, chunks=chunks)
    else:
        ds_ref = xr.open_dataset(ref_file)

    # Rename ckpt_ensemble -> number in the reference dataset if exists
    if "ckpt_ensemble" in ds_ref.dims:
        ds_ref = ds_ref.rename({"ckpt_ensemble": "number"})
    
    # Make sure the variable is in ds_ref
    if variable not in ds_ref.data_vars:
        ds_ref.close()
        raise ValueError(f"Variable '{variable}' not found in {ref_file}. "
                         f"Available vars: {list(ds_ref.data_vars.keys())}")

    # Build the set of (month, day) pairs from the reference dataset’s time
    unique_month_days = []
    for t_val in ds_ref.time.values:
        t_dt = xr.DataArray([t_val], dims="time").dt
        mm = int(t_dt.month.values)
        dd = int(t_dt.day.values)
        md = (mm, dd)
        if md not in unique_month_days:
            unique_month_days.append(md)

    # We no longer need ds_ref
    ds_ref.close()
    del ds_ref
    gc.collect()  # Force Python to release memory

    # ---------------------------------------------------------
    # 2) For each (month, day), gather data from all years
    # ---------------------------------------------------------
    for (mm, dd) in unique_month_days:
        # We'll accumulate each year's data in a list
        slices_for_all_years = []

        for year in years:
            # Path to that year’s hindcast
            year_file = os.path.join(source_dir, f"hindcast_{year}", "forecast_atmos.nc")
            if not os.path.isfile(year_file):
                # Skip if the file doesn't exist
                continue
            
            # Open with chunking if provided
            if chunks is not None:
                ds_year = xr.open_dataset(year_file, chunks=chunks)
            else:
                ds_year = xr.open_dataset(year_file)

            # Rename ckpt_ensemble -> number if it exists
            if "ckpt_ensemble" in ds_year.dims:
                ds_year = ds_year.rename({"ckpt_ensemble": "number"})
            
            # Ensure the variable is present
            if variable not in ds_year.data_vars:
                ds_year.close()
                continue
            
            # ---------------------------------------------------------
            # 2a) Select the data where time.month == mm and time.day == dd
            # ---------------------------------------------------------
            data_sel = ds_year[variable].where(
                (ds_year.time.dt.month == mm) & (ds_year.time.dt.day == dd),
                drop=True
            )

            # If there's no matching time, skip
            if data_sel.time.size == 0:
                ds_year.close()
                continue

            # If we find more than one time for the same day, raise an error
            if data_sel.time.size > 1:
                ds_year.close()
                raise ValueError(
                    f"Found multiple times for year={year}, month={mm}, day={dd}. "
                    "We assume exactly one time per day. Please verify your data."
                )
            
            # ---------------------------------------------------------
            # 2b) Overwrite the 'time' coordinate so it reflects the actual year
            # ---------------------------------------------------------
            orig_date = data_sel.time.values[0]
            new_time = np.datetime64(f"{year:04d}-{mm:02d}-{dd:02d}")
            data_sel = data_sel.assign_coords(time=("time", [new_time]))

            # Append the slice to the list
            slices_for_all_years.append(data_sel)
            
            # Close ds_year right after we get the slice
            ds_year.close()
            del ds_year
            gc.collect()

        # If no data was found for this month-day, skip
        if not slices_for_all_years:
            continue
        
        # ---------------------------------------------------------
        # 3) Concatenate along 'time' dimension
        # ---------------------------------------------------------
        combined = xr.concat(slices_for_all_years, dim="time")

        # ---------------------------------------------------------
        # 4) Save to a new file: /{output_dir}/{variable}/hindcast_{mm}-{dd}.nc
        # ---------------------------------------------------------
        out_file = os.path.join(var_output_path, f"hindcast_{mm:02d}-{dd:02d}.nc")
        print(f"Writing {out_file} ...")
        combined.to_netcdf(out_file)

        # Close and delete the combined data, and collect garbage
        combined.close()
        del combined
        # Also free the slices from memory
        slices_for_all_years.clear()
        gc.collect()

    print("Done restructuring hindcasts.")


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    source_directory = "/pscratch/sd/y/yacohen/nvdlesm/ensemble_atmos_98_good_ocean_Jan15"
    output_directory = "/pscratch/sd/y/yacohen/nvdlesm/ensemble_atmos_98_good_ocean_Jan15/reformatted"

    # Try chunking to avoid loading huge data into memory at once.
    # Adjust chunk sizes as appropriate for your data.
    chunk_dict = {'time': 1, 'height': 64, 'width': 64, 'face': 12}

    for var in ["t2m", "t850", "z500"]:
        restructure_hindcast(
            source_dir=source_directory,
            output_dir=output_directory,
            years=range(1996, 2016),
            variable=var,
            reference_year=1996,
            chunks=chunk_dict
        )
