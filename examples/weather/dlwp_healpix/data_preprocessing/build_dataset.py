# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import zarr
import datetime
import pandas as pd
import argparse
from apache_beam.options.pipeline_options import PipelineOptions

from earth2studio.data import ARCO, build_dataset
import earth2grid
import pandas as pd


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Build dataset for a specific year")
parser.add_argument("--year", type=int, required=True, help="Year to process")
parser.add_argument("--path", type=str, required=True, help="path to store the zarr data")
parser.add_argument("--level", type=int, default=6, help="Healpix resolution level to use (hpx<2**level>)")
args = parser.parse_args()

# variables and constants to download
variables = ["tcwv", "z1000", "z850", "z700", "z500", "z300", "z250", "z50", "t2m", "t850", "t500", "t250", "t50", "v10m", "v850", "v500", "v250", "v50", "u10m", "u850", "u500", "u250", "u50", "q850", "q500", "sst"]
constants = ["land_sea_mask", "z"]

# Set up date range for the entire year
start_date = f"{args.year}-01-01"
end_date = f"{args.year+1}-01-01"  # This will give us the full year, including Dec 31 23:00
time = pd.date_range(start=start_date, end=end_date, freq="3H", inclusive="left")
time_periods = len(time)

level = args.level # = hpx<2**level> 6 == 64 8 == 256
nside = 2**level

# where the output should be stored
storage_location = os.path.join(args.path,f"era5_hpx{nside}_{args.year}.zarr")

# Zarr cache location
zarr_cache = zarr.open(os.path.join(args.path,"my_zarr_cache.zarr"), mode="a")
# Create the data source
data = ARCO(zarr_cache=zarr_cache)

# set the grid, healpix has HEALPIX_PAD_XY, and HEALPIX_XY() as options
hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
src = earth2grid.latlon.equiangular_lat_lon_grid(721, 1440)
regrid = earth2grid.get_regridder(src, hpx)
hw = [i for i in range(nside)]
face = [i for i in range(12)]

def _transform(regrid, nside):
    import torch
    def transform(time, variable, x):
        x_torch = torch.as_tensor(x)
        x_torch = x_torch.double()
        hlp_x = regrid(x_torch).reshape(12, nside, nside)
        return hlp_x.numpy()
    return transform
transform = _transform(regrid, nside)

# Apache Beam options
options = PipelineOptions([
    '--runner=DirectRunner',
    '--direct_num_workers=2',
    '--direct_running_mode=multi_processing',
])

# Make Store Dataset
start_time = datetime.datetime.now()
print(f'{start_date} getting {time_periods} samples starting at {start_date}')
time = pd.date_range(start_date, freq="3h", periods=time_periods)
zarr_dataset = zarr.open(storage_location, mode="w")

# Build the predicted variables
print("\ngetting predicted variables\n")
dataset = zarr_dataset.create_dataset("inputs", shape=(time_periods, len(variables), 12, nside, nside), chunks=(1, len(variables), 12, nside, nside), dtype="f4")
build_dataset(data, time, variables, dataset, apache_beam_options=options, transform=transform)

print("\ngetting target variables\n")
dataset = zarr_dataset.create_dataset("targets", shape=(time_periods, len(variables), 12, nside, nside), chunks=(1, len(variables), 12, nside, nside), dtype="f4")
build_dataset(data, time, variables, dataset, apache_beam_options=options, transform=transform)

# Build the un-predicted variables
print("\ngetting constants\n")
dataset = zarr_dataset.create_dataset("channel_c", shape=(1, len(constants), 12, nside, nside), chunks=(1, len(constants), 1, nside, nside), dtype="f4")
build_dataset(data, [time[0]], constants, dataset, apache_beam_options=options, transform=transform)
dataset = zarr_dataset.create_dataset("constants", shape=(len(constants), 12, nside, nside), chunks=(len(constants), 1, nside, nside), dtype="f4")
dataset[:,:,:,:] = zarr_dataset["channel_c"][0,:,:,:]

print("\nSetting up attributes\n")

zarr_dataset.create_dataset("time", data=time.to_numpy(), overwrite=True)
zarr_dataset.create_dataset("face", data=face, overwrite=True, dtype="int64", fill_value=None)
zarr_dataset.create_dataset("width", data=hw, overwrite=True, dtype="int64", fill_value=None)
zarr_dataset.create_dataset("height", data=hw, overwrite=True, dtype="int64", fill_value=None)
zarr_dataset.create_dataset("channel_in", data=variables, overwrite=True, dtype="<U20")
zarr_dataset.create_dataset("channel_out", data=variables, overwrite=True, dtype="<U20")
zarr_dataset.create_dataset("channel_c", data=constants, overwrite=True, dtype="<U20")
zarr_dataset.create_dataset("level", shape=(), dtype="f4")

# Set attributes
zarr_dataset["face"].attrs["_ARRAY_DIMENSIONS"] = ["face"]
zarr_dataset["width"].attrs["_ARRAY_DIMENSIONS"] = ["width"]
zarr_dataset["height"].attrs["_ARRAY_DIMENSIONS"] = ["height"]
zarr_dataset["channel_in"].attrs["_ARRAY_DIMENSIONS"] = ["channel_in"]
zarr_dataset["channel_out"].attrs["_ARRAY_DIMENSIONS"] = ["channel_out"]
zarr_dataset["channel_c"].attrs["_ARRAY_DIMENSIONS"] = ["channel_c"]
zarr_dataset["level"].attrs["_ARRAY_DIMENSIONS"] = []
zarr_dataset["level"].attrs["axis"] = ["Z"]

zarr_dataset["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
zarr_dataset["time"].attrs["axis"] = ["T"]
zarr_dataset["time"].attrs["calendar"] = ["gregorian"]
zarr_dataset["time"].attrs["long_name"] = ["time"]
zarr_dataset["time"].attrs["standard_name"] = ["time"]
zarr_dataset["time"].attrs["units"] = ["hours since 1900-01-01"]

zarr_dataset["inputs"].attrs["_ARRAY_DIMENSIONS"] = ["time", "channel_in", "face", "height", "width"]
zarr_dataset["targets"].attrs["_ARRAY_DIMENSIONS"] = ["time", "channel_out", "face", "height", "width"]
zarr_dataset["constants"].attrs["_ARRAY_DIMENSIONS"] = ["channel_c", "face", "height", "width"]

lon_r = hpx.lon.reshape(12, nside, nside)
lat_r = hpx.lat.reshape(12, nside, nside)

zarr_dataset.create_dataset("lat", data=lat_r, shape=(12,nside,nside), overwrite=True, fill_value=float("NaN"))
zarr_dataset.create_dataset("lon", data=lon_r, shape=(12,nside,nside), overwrite=True, fill_value=float("NaN"))

zarr_dataset["lat"].attrs["_ARRAY_DIMENSIONS"] = ["face", "height", "width"]
zarr_dataset["lon"].attrs["_ARRAY_DIMENSIONS"] = ["face", "height", "width"]

zarr_dataset["inputs"].attrs["coordinates"] = "level lon lat"
zarr_dataset["targets"].attrs["coordinates"]  = "level lon lat"
zarr_dataset["constants"].attrs["coordinates"] = "level lon lat"

# create consolidated metadata to reduce loading time
zarr.consolidate_metadata(storage_location)

end_time = datetime.datetime.now()
print(f"{end_time} done")
elapsed = (end_time - start_time)
print(f"elapsed {elapsed}")

# Update the checkpoint file
with open("processed_years.log", "a") as f:
    f.write(f"{args.year}\n")
