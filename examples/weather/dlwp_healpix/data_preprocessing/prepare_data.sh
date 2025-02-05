#!/bin/bash

# Set the range of years
START_YEAR=1980
END_YEAR=2023

# Path to the checkpointing log file
CHECKPOINT_FILE="processed_years.log"

if [ "$#" -ne 2 ]; then
    echo "Usage $0 </path/to/yearly/zarr/files> <level>"
    exit
fi

# Create the output directory if it doesn't exist
mkdir -p ${1}

# Create checkpoint file if it doesn't exist
touch ${1}/$CHECKPOINT_FILE

# this loop is inclusive 
for year in $(seq $START_YEAR $END_YEAR); do
    # Check if the year has already been processed
    if grep -q "^$year$" "$1/$CHECKPOINT_FILE"; then
        echo "Year $year already processed. Skipping."
        continue
    fi

    echo "Processing year $year"
    sbatch submit_nvdlesm_data.sh ${year} ${1} ${2}

    # Wait for a short time before submitting the next job
    sleep 5
done

echo "All jobs submitted. Check the Slurm queue for progress."
