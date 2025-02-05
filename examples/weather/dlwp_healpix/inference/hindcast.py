import os
import subprocess
import json
import argparse
import glob

def load_config(config_path):
    """
    Load the JSON configuration file.
    Args:
        config_path (str): Path to the JSON configuration file.
    Returns:
        dict: Configuration data as a dictionary.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def is_run_completed(output_dir, config):
    """
    Check if a run is actually completed by verifying both the marker file
    and the expected output files exist.
    """
    # Check for completion marker
    if not os.path.exists(os.path.join(output_dir, 'completed.txt')):
        return False
        
    # Check for actual output files
    atmos_pattern = f"{os.path.join(output_dir, config['atmos_output_filename'])}*.nc"
    ocean_pattern = f"{os.path.join(output_dir, config['ocean_output_filename'])}*.nc"
    
    atmos_files = glob.glob(atmos_pattern)
    ocean_files = glob.glob(ocean_pattern)
    
    if not (atmos_files and ocean_files):
        # If we find a completed.txt but no output files, remove the marker
        os.remove(os.path.join(output_dir, 'completed.txt'))
        return False
        
    return True

def create_inference_config(base_config, year):
    """
    Create a configuration for a single inference run.
    """
    inference_config = base_config.copy()
    inference_config['forecast_init_start'] = f"{year}-01-02"
    inference_config['forecast_init_end'] = f"{year}-12-30"
    inference_config['output_directory'] = os.path.join(base_config['base_output_dir'], f"hindcast_{year}")
    return inference_config

def run_inference(inference_config):
    """
    Run a single inference for a specific year.
    coupled_forecast.py will handle GPU parallelization internally.
    """
    output_dir = inference_config['output_directory']
    
    if is_run_completed(output_dir, inference_config):
        print(f"Run for year {inference_config['forecast_init_start'][:4]} is already completed (verified outputs exist), skipping...")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove any existing completed.txt if present
    completion_marker = os.path.join(output_dir, 'completed.txt')
    if os.path.exists(completion_marker):
        os.remove(completion_marker)
    
    temp_config_path = os.path.join(output_dir, 'temp_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(inference_config, f, indent=2)
    
    command = [
        "python", "-W", "ignore", "coupled_forecast.py",
        "--config", temp_config_path
    ]
    
    print(f"Running forecast for year {inference_config['forecast_init_start'][:4]}")
    try:
        result = subprocess.run(command, check=True)
        
        # Only write completion marker if subprocess was successful
        # and output files exist
        if result.returncode == 0 and is_run_completed(output_dir, inference_config):
            with open(completion_marker, 'w') as f:
                f.write("Run completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running forecast for year {inference_config['forecast_init_start'][:4]}: {e}")
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def run_hindcast(config):
    """
    Run hindcast simulations sequentially for multiple years.
    coupled_forecast.py will handle GPU parallelization internally.
    """
    if args.start_year is not None and args.end_year is not None:
        start_year = args.start_year
        end_year = args.end_year
    else:
        start_year = config.get('start_year', 1996)
        end_year = config.get('end_year', 2016)
    
    years = list(range(start_year, end_year + 1))
    
    for year in years:
        inference_config = create_inference_config(config, year)
        run_inference(inference_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hindcast simulations")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--start-year", type=int, help="Start year for hindcast (optional)")
    parser.add_argument("--end-year", type=int, help="End year for hindcast (optional)")
    
    args = parser.parse_args()
    config = load_config(args.config)

    # If --start-year is provided, overwrite the config entry
    if args.start_year is not None:
        config["start_year"] = args.start_year

    # If --end-year is provided, overwrite the config entry
    if args.end_year is not None:
        config["end_year"] = args.end_year

    # Now call run_hindcast with the updated (or unmodified) config
    run_hindcast(config)