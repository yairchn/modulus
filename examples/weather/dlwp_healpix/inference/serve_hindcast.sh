#!/bin/bash
# serve_hindcast.sh
# Usage: bash serve_hindcast.sh <number_of_jobs>

if [ $# -ne 1 ]; then
    echo "Usage: bash serve_hindcast.sh <number_of_jobs>"
    exit 1
fi

START_YEAR=1996
END_YEAR=2016
TOTAL_YEARS=$((END_YEAR - START_YEAR))
NUM_JOBS=$1

# Validate even division
if [ $((TOTAL_YEARS % NUM_JOBS)) -ne 0 ]; then
    echo "Error: Number of jobs ($NUM_JOBS) must evenly divide total years ($TOTAL_YEARS)"
    exit 1
fi

YEARS_PER_JOB=$((TOTAL_YEARS / NUM_JOBS))
current_year=$START_YEAR

# Submit jobs
for ((job=1; job<=NUM_JOBS; job++)); do
    job_end_year=$((current_year + YEARS_PER_JOB - 1))
    echo "Submitting job ${job} for years ${current_year}-${job_end_year}"
    sbatch submit_hindcast.sh ${current_year} ${job_end_year}
    current_year=$((job_end_year + 1))
done

echo "Submitted $NUM_JOBS jobs"