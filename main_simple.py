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

def main():
    args = parse_args()
    
    # Load the main module
    # module = importlib.import_module(f'engines.{args.model}')
    module = getattr(engines, args.model, None)
    device = read_device(args.device)

    if not module:
        raise ModuleNotFoundError(f"Module {args.model} not found")

    if not isinstance(module, types.ModuleType):
        raise TypeError(f"{args.model} is not a module")

    # Load the configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get log directory and version
    log_dir = config['train']['log_dir']
    version = get_version(log_dir)

    module.train(config=config.copy(), device=device)

if __name__ == '__main__':
    main()