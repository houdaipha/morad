import os
import argparse
import types
import torch
import yaml
import signal
import logging
import importlib
import engines
from engines.utils import copy_logs, get_version

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        help='path to config file',
        required=True)
    parser.add_argument('-m', '--model', help='Model name', required=True)
    parser.add_argument(
        '-d',
        '--device',
        default="cpu",
        help='cuda device: cpu, integer, or list of integers separated by comma',
        type=str)
    parser.add_argument(
        '-t',
        '--temp_dir',
        help='Temporary directory to save logs',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    return args

# Mitigate gpu allocation problem
# NOTE: Should remove before the final version
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def read_device(device):
    if device != 'cpu':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available')
        device = [int(item) for item in device.split(',')]
    return device

def get_temp_dir(model):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(dir_name, 'temp_logs', model)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def main():
    args = parse_args()
    
    # Load the main module
    # module = importlib.import_module(f'engines.{args.model}')
    module = getattr(engines, args.model, None)
    device = read_device(args.device)

    if not module:
        raise ModuleNotFoundError(f"Module engines.{args.model} not found")

    if not isinstance(module, types.ModuleType):
        raise TypeError(f"engines.{args.model} is not a module")

    # Handle temporary directory
    if args.temp_dir:
        temp_dir = get_temp_dir(args.model)
    else:
        temp_dir = None

    # Load the configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get log directory and version
    log_dir = config['train']['log_dir']
    version = get_version(log_dir)

    # Handle exit signals
    copy_logs_called = False
    def handle_exit(signum, frame):
        nonlocal copy_logs_called
        if not copy_logs_called and args.temp_dir:
            logger.info("Signal received. Initiating copy before exit.")
            copy_logs(log_dir, temp_dir, version, remove_weights=True)
            copy_logs_called = True
        exit(0)
    signal.signal(signal.SIGINT, handle_exit)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit) # Handle kill signal

    try:
        module.train(config=config.copy(), device=device, temp_dir=temp_dir)
    finally:
        if not copy_logs_called and args.temp_dir:
            logger.info("Training finished. Initiating copy before exit.")
            copy_logs(log_dir, temp_dir, version, remove_weights=True)
            copy_logs_called = True

if __name__ == '__main__':
    main()