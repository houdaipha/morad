import os
import types
import logging
import argparse

import torch
from sklearn.metrics import classification_report

from training.utils import prediction_results
import engines

DEFAULT_PATH = '/morad_dir/weights/MORAD/TEST'


def parse_args():
    parser = argparse.ArgumentParser(description='Testing module performance')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        help='Model name',
        required=True)
    parser.add_argument(
        '--version',
        '-v',
        type=int,
        help='Model version',
        required=True)
    parser.add_argument(
        '--path',
        '-p',
        default=DEFAULT_PATH,
        type=str,
        help='Path to the model')
    parser.add_argument(
        '--split',
        '-s',
        default='test',
        type=str,
        help='Split to test on')
    parser.add_argument(
        '--device', '-d', default='cpu', type=str, help='Device to use')
    parser.add_argument(
        '--verbose', '-V', action='store_true', help='Verbose mode')
    return parser.parse_args()

def read_device(device):
    if device != 'cpu':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available')
        device = [int(item) for item in device.split(',')]
        if len(device) > 1:
            print('Multiple GPUs detected, using the first one')
        device = f'cuda:{device[0]}'
    return torch.device(device)


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load the main module
    module = getattr(engines, args.model)
    if not module:
        raise ModuleNotFoundError(f"Module engines.{args.model} not found")

    if not isinstance(module, types.ModuleType):
        raise TypeError(f"engines.{args.model} is not a module")

    device = read_device(args.device)

    ground_truth, predections, _ = module.test(
        path=os.path.join(args.path, args.model, f'version_{args.version}'),
        device=device,
        split=args.split,
        load_best=True)

    print(classification_report(ground_truth, predections))

    print(f'Model: {args.model}, Version: {args.version}')
    print(prediction_results(predections, ground_truth))

if __name__ == '__main__':
    main()