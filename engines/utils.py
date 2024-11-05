import sys
import os
import shutil
import logging
import random
import numpy as np
import torch
import subprocess
from time import sleep

logger = logging.getLogger(__name__)

def get_log_dir(path):
    # Create directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)
        logger.info(f'Created directory: {path}')

        log_dir = os.path.join(path, 'version_0')
        os.mkdir(log_dir)
        return log_dir
    
    # Get the last version
    versions = [
        int(v.split('_')[-1]) for v in os.listdir(path) if v.startswith('version')]
    last_version = max(versions) + 1 if versions else 0

    log_dir = os.path.join(path, f'version_{last_version}')
    os.mkdir(log_dir)
    return log_dir

def get_version(path):
    # Create directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)
        logger.info(f'Created directory: {path}')

        # log_dir = os.path.join(path, 'version_0')
        # os.mkdir(log_dir)
        return 0
    
    # Get the last version
    versions = [
        int(v.split('_')[-1]) for v in os.listdir(path) if v.startswith('version')]
    last_version = max(versions) + 1 if versions else 0

    # log_dir = os.path.join(path, f'version_{last_version}')
    # os.mkdir(log_dir)
    return last_version

def copy_logs(log_dir, temp_dir, version, remove_weights=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_dir = os.path.join(log_dir, f'version_{version}')
    
    if not os.path.exists(log_dir):
        logger.info('No logs to copy, destination directory does not exist')
        return False

    temp_dir = os.path.join(temp_dir, f'version_{version}')

    if not os.path.exists(temp_dir):
        logger.info('No logs to copy, source directory does not exist')
        return False

    logger.info(f'Copying logs from {temp_dir} to {log_dir}')

    # Use rsync to copy logs from temp_dir to log_dir
    try:
        logger.info('Copying logs with rsync')
        logger.info(f'rsync -av {temp_dir}/ {log_dir}')
        subprocess.run(
            ["rsync", "-av", f"{temp_dir}/", log_dir],
            check=True
        )
        logger.info('Logs copied successfully with rsync')
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to copy logs with rsync: {e}')
        return False

    # Remove weights if specified
    if remove_weights:
        try:
            weights_dir = os.path.join(temp_dir, 'weights')
            if os.path.exists(weights_dir):
                logger.info('Removing weights')
                logger.info(f'rm -rf {weights_dir}')
                subprocess.run(
                    ["rm", "-rf", weights_dir],
                    check=True
                )
                logger.info('Weights removed')
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to remove weights: {e}')
            return False
    return True

def config_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, "w", "utf-8"))
        print("Logging to {}".format(log_file))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        force=True
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # Seed all CUDA devices
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def handle_device(device):
    if device != 'cpu':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available')
        if not isinstance(device, list):
            raise ValueError('Device must be a list of integers')
        return f'cuda:{device[0]}'
    return device