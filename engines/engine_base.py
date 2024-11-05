import os
import logging
import json
import yaml
from dataclasses import asdict
from typing import Type, Callable

import torch
from engines.utils import get_version, config_logging, set_seed, handle_device

logger = logging.getLogger(__name__)

class Engine:
    def __init__(
        self,
        data_config_cls: Type,
        data_manager_cls: Type,
        model_config_cls: Type,
        model_cls: Type,
        train_config_cls: Type,
        pipeline: Callable,
    ):
        self.data_config_cls = data_config_cls
        self.data_manager_cls = data_manager_cls
        self.model_config_cls = model_config_cls
        self.model_cls = model_cls
        self.train_config_cls = train_config_cls
        self.pipeline = pipeline


    def train(self, config, device, temp_dir=None):
        main_device = handle_device(device)

        # Get log directory
        version = get_version(config['train']['log_dir'])
        log_dir = os.path.join(config['train']['log_dir'], f'version_{version}')
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f'Log directory: {log_dir}')

        # Handle temporary directory if provided
        main_log_dir = None
        if temp_dir is not None:
            main_log_dir = log_dir
            log_dir = os.path.join(temp_dir, f'version_{version}')
            os.makedirs(log_dir, exist_ok=True)

        # Update config with runtime values
        config['train']['version'] = version
        config['train']['device'] = main_device
        config['train']['log_dir'] = log_dir

        # Configure logging
        config_logging(log_file=os.path.join(log_dir, 'train.log'))

        # Create configurations
        train_config = self.train_config_cls(**config['train'])
        model_config = self.model_config_cls(**config['model'])
        data_config = self.data_config_cls(**config['data'])
        logger.info('Configs loaded')

        # Handle logging
        log_train_config = asdict(train_config)
        if temp_dir is not None:
            log_train_config['log_dir'] = main_log_dir

        log_config = {
            'data': asdict(data_config),
            'model': asdict(model_config),
            'train': log_train_config,
            'notes': config.get('notes', '')
        }
        
        logger.info('Configurations:')
        logger.info(f'{json.dumps(log_config, indent=4)}')

        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(log_config, f)

        if train_config.seed is not None:
            set_seed(train_config.seed)
            logger.info(f'Seed set to {train_config.seed}')

        # Initialize model and data
        model = self.model_cls(model_config, main_device)
        data = self.data_manager_cls(data_config)
        logger.info('Model and data initialized')

        # Handle multi-GPU setup
        if isinstance(device, list) and len(device) > 1:
            model = nn.DataParallel(model, device)
            logger.info(f'Training using DataParallel on devices {device}')

        # Train
        self.pipeline.fit(train_config, model, data, log_config=log_config)

        logger.info('Training finished')
        return model

    def test(self, path, device, split='test', epoch=None, load_best=False):
        # Get checkpoint path
        if epoch is None and not load_best:
            raise ValueError(
                'Specify either an epoch or set load_best=True, not both.')
        if load_best:
            checkpoint_name = 'best_loss.pth'
        elif epoch is not None:
            checkpoint_name = f'checkpoint_{epoch}.pth'
        else:
            raise ValueError('Specify either an epoch or set load_best=True')

        checkpoint_path = os.path.join(
            path, 'weights', checkpoint_name)
        logger.info(f'Using checkpoint from: {checkpoint_path}')

        if not isinstance(device, torch.device):
            raise ValueError('Device should be a torch.device object')

        logger.info(f'Using device: {device}')

        with open(os.path.join(path, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)


        model_config = self.model_config_cls(**config['model'])
        data_config = self.data_config_cls(**config['data'])

        model = self.model_cls(model_config, device)
        data = self.data_manager_cls(data_config)

        return self.pipeline.test(model, data, checkpoint_path, device, split)