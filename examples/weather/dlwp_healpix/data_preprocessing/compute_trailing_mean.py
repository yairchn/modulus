import zarr
import numpy as np
import pandas as pd
import argparse
import os
from glob import glob


def compute_trailing_mean(
    file_path, variables_name=["ws10m"], coupled_dt="6h", influence_window="48h"
):
    print(f"Processing {file_path}")
    ds = zarr.open(file_path, mode="r+")

    channel_in = ds.channel_in[:]
    channel_out = ds.channel_out[:]
    time = ds.time[:]
    data_dt = time[1] - time[0]
    coupled_dt = pd.Timedelta(coupled_dt)
    ratio_dt = int(coupled_dt // data_dt)

    for variable_name in variables_name:
        if variable_name not in channel_in:
            print(f"{variable_name} not exits")
            continue

        variable_index = np.where(channel_in == variable_name)[0][0]
        variable_data = ds.inputs[:, variable_index]

        variable_average_data = np.zeros_like(variable_data)

        influence_window_pd = pd.Timedelta(influence_window)
        influence_window_ind = int(influence_window_pd // data_dt)
        first_valid_sample_index = np.where(time == time[0] + influence_window_pd)[0][0]

        for i in range(first_valid_sample_index, len(time)):
            if i<first_valid_sample_index:
                variable_average_data[i] = variable_data[i]
            else:
                variable_average_data[i] = variable_data[
                    (i - influence_window_ind) : (i + 1) : ratio_dt
                ].mean(axis=0)

        new_shape = list(ds.inputs.shape)
        new_shape[1] += 1
        ds.inputs.resize(*new_shape)
        ds.targets.resize(*new_shape)

        # Add trailing mean to inputs and targets
        ds.inputs[:, -1] = variable_average_data
        ds.targets[:, -1] = variable_average_data

        variable_name_average = variable_name + "-" + influence_window.upper()

        # Update channel lists
        channel_in = np.append(channel_in, variable_name_average)
        channel_out = np.append(channel_out, variable_name_average)

    # Update channel_in and channel_out datasets
    ds.channel_in.resize(len(channel_in))
    ds.channel_in[:] = channel_in
    ds.channel_out.resize(len(channel_out))
    ds.channel_out[:] = channel_out

    # Consolidate metadata
    zarr.consolidate_metadata(file_path)
    print(f"Finished processing {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add derived channels to yearly Zarr files."
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        required=True,
        help="Path to the folder containing yearly Zarr files",
    )
    parser.add_argument(
        "--variables_name", nargs="+", default=[], help="name of variable in dataset"
    )
    parser.add_argument(
        "--coupled_dt",
        type=str,
        default="6h",
        help="the time resoluton of the atmos model to be coupled",
    )
    parser.add_argument(
        "--influence_window", type=str, default="48h", help="range for averaging "
    )
    args = parser.parse_args()

    # Get all Zarr files in the target folder
    zarr_files = sorted(glob(os.path.join(args.target_folder, "*.zarr")))

    for zarr_file in sorted(zarr_files):
        compute_trailing_mean(
            zarr_file,
            variables_name=args.variables_name,
            coupled_dt=args.coupled_dt,
            influence_window=args.influence_window,
        )


if __name__ == "__main__":
    main()
