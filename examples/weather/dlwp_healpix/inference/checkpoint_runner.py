import os
import subprocess
import json
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
import torch
import random
import socket
import random


def find_free_port(start=29500, end=29999):
    while True:
        port = random.randint(start, end)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port  # Port is free
            except OSError:
                continue  # Port is already in use


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

def check_output_files_exist(output_dir, config):
    """
    Check if the expected output files exist, without checking for marker.
    """
    atmos_pattern = f"{os.path.join(output_dir, config['atmos_output_filename'])}*.nc"
    ocean_pattern = f"{os.path.join(output_dir, config['ocean_output_filename'])}*.nc"
    
    atmos_files = glob.glob(atmos_pattern)
    ocean_files = glob.glob(ocean_pattern)
    
    return bool(atmos_files and ocean_files)

def is_run_completed(output_dir, config):
    """
    Check if a run is actually completed by verifying both the marker file
    and the expected output files exist.
    """
    if not os.path.exists(os.path.join(output_dir, 'completed.txt')):
        # If output files exist but no marker, create the marker
        if check_output_files_exist(output_dir, config):
            with open(os.path.join(output_dir, 'completed.txt'), 'w') as f:
                f.write("Run completed successfully")
            return True
        return False
        
    # If marker exists but no outputs, remove marker
    if not check_output_files_exist(output_dir, config):
        os.remove(os.path.join(output_dir, 'completed.txt'))
        return False
        
    return True

def create_checkpoint_config(base_config, atmos_ckpt, ocean_ckpt, ckpt_index, gpu_id):
    """
    Create a configuration for a single checkpoint combination run.
    """
    checkpoint_config = base_config.copy()
    
    # Set single checkpoints instead of lists
    checkpoint_config['atmos_model_checkpoint'] = atmos_ckpt
    checkpoint_config['ocean_model_checkpoint'] = ocean_ckpt
    
    # Create a subdirectory for this checkpoint combination
    base_output_dir = checkpoint_config['output_directory']
    checkpoint_config['output_directory'] = os.path.join(base_output_dir, f"ckpt_{ckpt_index}")
    
    # Set specific GPU
    checkpoint_config['gpu'] = 0  # Each run uses one GPU
    
    return checkpoint_config

def run_forecast(args):
    """
    Run a single forecast for a specific checkpoint combination on a specific GPU.
    """
    checkpoint_config, gpu_id = args
    output_dir = checkpoint_config['output_directory']
    
    if is_run_completed(output_dir, checkpoint_config):
        print(f"Run for checkpoint in directory {output_dir} is already completed (verified outputs exist), skipping...")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove any existing completed.txt if present
    completion_marker = os.path.join(output_dir, 'completed.txt')
    if os.path.exists(completion_marker):
        os.remove(completion_marker)
    
    # Remove any existing config.json if present
    temp_config_path = os.path.join(output_dir, 'temp_config.json')
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    with open(temp_config_path, 'w') as f:
        json.dump(checkpoint_config, f, indent=2)
    
    # Get a random port in the range 29500-29999
    port = find_free_port()
    
    # Set environment variables for GPU and distributed training
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['MASTER_PORT'] = str(port)  # Dynamically set the port
    env['MASTER_ADDR'] = '127.0.0.1'  # Use localhost instead of 0.0.0.0
    env['WORLD_SIZE'] = '1'
    env['RANK'] = '0'
    env['LOCAL_RANK'] = '0'
    
    command = [
        "python", "-W", "ignore", "coupled_forecast.py",
        "--config", temp_config_path
    ]
    
    print(f"Running forecast for checkpoint in {output_dir} on GPU {gpu_id}, using port {port}")
    try:
        result = subprocess.run(command, check=True, env=env)
        
        # Check if run was successful and files were created
        if result.returncode == 0 and check_output_files_exist(output_dir, checkpoint_config):
            with open(completion_marker, 'w') as f:
                f.write("Run completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running forecast for checkpoint in {output_dir} on GPU {gpu_id}: {e}")


def get_available_gpu_count():
    """
    Get the number of available GPUs on the system.
    """
    try:
        return torch.cuda.device_count()
    except:
        return 0

def run_multi_checkpoint_forecast(config, max_parallel_runs=None):
    """
    Run forecasts for all combinations of atmosphere and ocean checkpoints
    in parallel across available GPUs.
    """
    atmos_checkpoints = config['atmos_model_checkpoints']
    ocean_checkpoints = config['ocean_model_checkpoints']
    
    # Get number of available GPUs
    num_gpus = get_available_gpu_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available on this system")
    
    if max_parallel_runs is None:
        max_parallel_runs = num_gpus
    else:
        max_parallel_runs = min(max_parallel_runs, num_gpus)
    
    print(f"Using {max_parallel_runs} GPUs for parallel execution")
    
    # Prepare all checkpoint combinations with their assigned GPUs
    run_configs = []
    ckpt_index = 0
    for atmos_ckpt in atmos_checkpoints:
        for ocean_ckpt in ocean_checkpoints:
            gpu_id = ckpt_index % max_parallel_runs
            checkpoint_config = create_checkpoint_config(config, atmos_ckpt, ocean_ckpt, ckpt_index, gpu_id)
            run_configs.append((checkpoint_config, gpu_id))
            ckpt_index += 1
    
    # Run forecasts in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_parallel_runs) as executor:
        list(executor.map(run_forecast, run_configs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-checkpoint forecasts')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--max-gpus', type=int, help='Maximum number of GPUs to use in parallel (default: all available)')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    run_multi_checkpoint_forecast(config, args.max_gpus)
