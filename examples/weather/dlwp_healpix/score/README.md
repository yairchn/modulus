# ML Weather Model Scoring

This directory contains code for scoring and bias computation of the ML weather model.

## Overview

The codebase includes two main components:
1. Model scoring and evaluation metrics
2. Bias computation and storage

## Scoring Components

- `rmse_acc.py`: Legacy implementation of RMSE and ACC metrics
- `score_coupled_forecast.py`: Scoring code for deterministic and probabelistic scores using xskillscore
- `plot_bias_maps.py`: Visualization of bias distributions
- `save_model_bias.py`: compute the bias across provided hindcast years. The hindcast is expect as a folder with a file per month_year combination acorss all hindcast years. See inference/REAME.md
- `score_s2.py`: S2S forecast evaluation, with bias corrections, uses score_medium_range requires a bias file produced with `save_model_bias.py`. Here you can set the averaging window for the bias (like 28 days) and the averaging window for the forecast (like 7 days)
- `plot_scores.py`: Plots the metrics computed by `score_coupled_forecast.py`
- `postprocessing.py`: Post-processing utilities for model outputs

## Bias Computation

The bias computation pipeline is implemented in `save_model_bias.py`. This script:
- Takes a combined hindcast folder as input (see `inference/README.md` for details)
- Computes bias for all specified channels (defined in the code)
- Saves results to a NetCDF file
- Bias has both the interannual mean and interannual STD to allow us to asses the statistical segnificance of the bias. 

### Bias Output Format

The generated bias NetCDF file includes:
- Computed bias values for each channel
- Complete configuration of the hindcast
- All necessary information for reproducibility

## Usage

```bash
# Submit bias computation job
./submit_bias.sh
```

## Dependencies

- Python scripts for computation and analysis
- NetCDF support for data storage
- Shell scripts for job submission

For more details about specific components, please refer to the individual script files.
