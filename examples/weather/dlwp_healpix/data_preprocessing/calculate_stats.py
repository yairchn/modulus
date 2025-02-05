import argparse
import xarray as xr


def calculate_stats(source, output_file, calculate_climatology):
    ds = xr.open_zarr(source)

    channel_list = ds.channel_in[:]

    print("calculating mean")
    mean = ds.inputs.mean(dim=['time','face','height','width'], keep_attrs=True)
    print("calculating std")
    std = ds.inputs.std(dim=['time','face','height','width'], keep_attrs=True)

    mean.rename({"channel_in":"channel"})
    std.rename({"channel_in":"channel"})

    stats = mean.to_dataset(name="means")
    stats["stddev"] = std # to avoid a name collisions use stddev

    if calculate_climatology:
        print("Calculating climatology (this could take some time)")
        clima = ds.inputs.groupby('time.dayofyear').mean(dim='time', keep_attrs=True)
        clima.rename({"channel_in":"channel"})
        stats["climatology"] = clima

    for channel_num in range(len(stats.channel_in)):
        print(f"{stats.channel_in.values[channel_num]}:")
        print(f"  mean: {stats.means.values[channel_num]}")
        print(f"  std:  {stats.stddev.values[channel_num]}")

    stats.to_netcdf(output_file)
    print(f"Stats saved to {output_file}")



def main():
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation of a zarr dataset.")
    parser.add_argument("--source", type=str, required=True, help="The source zarr for which to calculate the stats")
    parser.add_argument("--output_file", type=str, default="stats.zarr", help="Where to store the statistics")
    parser.add_argument("--daily_climatology", action="store_true", help="Calculate a basic daily climatology")
    args = parser.parse_args()

    calculate_stats(args.source, args.output_file, args.daily_climatology)

if __name__ == "__main__":
    main()

