import os
import subprocess
import concurrent.futures

def run_combine_checkpoint(hindcast_dir, gpu_id):
    """
    Calls combine_checkpoints.py on hindcast_dir using the specified GPU.
    """
    # Copy current environment variables
    env = os.environ.copy()
    # Restrict visibility to the specified GPU
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        "python", 
        "combine_checkpoints.py", 
        "--directory", 
        hindcast_dir
    ]
    
    print(f"Starting: {cmd} on GPU {gpu_id}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Optionally print or log stdout/stderr
    if result.returncode == 0:
        print(f"Completed successfully: {hindcast_dir} on GPU {gpu_id}")
    else:
        print(f"Error in {hindcast_dir} on GPU {gpu_id}:\n{result.stderr}")

def main():
    base_path = "/lustre/fsw/coreai_climate_earth2/yacohen/nvdlesm/perl_copy/hindcast/HPX64/ensemble_atmos_98_good_ocean_Jan15_healpix_dir/"
    
    # skip_dirs = ['hindcast_1999', 'hindcast_2005'] # if you like to skip dirs
    skip_dirs = []

    # List all subdirectories in base_path that start with "hindcast_"
    subdirs = sorted(
        d for d in os.listdir(base_path) 
        if d.startswith("hindcast_") and os.path.isdir(os.path.join(base_path, d))
    )
    subdirs = [d for d in subdirs if d not in skip_dirs]
    
    # Number of GPUs on the node
    num_gpus = 8
    
    # Create a ThreadPool (or ProcessPool) to run tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        # Submit each hindcast directory to the executor
        futures = []
        for i, d in enumerate(subdirs):
            # Assign GPU in a round-robin fashion
            gpu_id = i % num_gpus
            hindcast_dir = os.path.join(base_path, d)
            futures.append(
                executor.submit(run_combine_checkpoint, hindcast_dir, gpu_id)
            )
        
        # Optionally wait for all tasks to complete and handle results
        for f in concurrent.futures.as_completed(futures):
            # This will raise any exceptions that occurred in the subprocess call
            f.result()

if __name__ == "__main__":
    main()
