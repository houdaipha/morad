from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type
from abc import ABC, abstractmethod
import os
import logging
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from training.utils import TriStageLRSchedulerLambda, FocalLoss
from training.utils import TrainClock, AverageMeter, prediction_results, format_config

logger = logging.getLogger(__name__)

ADAM_OPTIMIZER = {
    'name': 'Adam',
    'betas': [0.9, 0.999],
    'weight_decay': 0.01
}

SCHEDULER = {
    'final_decay': 'exp',
    'phase_ratio': [0.1, 0., 0.9]
}

CRITERION = {
    'name': 'cross_entropy'
}


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    batch_size: int
    log_dir: str
    version: int
    device: str = 'cpu'
    optimizer: dict = field(default_factory=lambda: ADAM_OPTIMIZER)
    scheduler: dict = field(default_factory=lambda: SCHEDULER)
    num_workers: int = 4
    pin_memory: bool = True
    debug: bool = False
    checkpoint_interval: int = 1
    seed: int | None = None
    oversample: bool = False
    criterion: str = field(default_factory=lambda: CRITERION)


class BaseTrainer:
    """A modular trainer class that handles training loop, validation, and testing."""

    def __init__(
        self,
        model: Type,
        config: Type,
        forward_fn: Callable
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            config: Training configuration
            data: data manager object
            forward_fn: Function that handles forward pass
        """
        self.config = config
        self.device = config.device
        self.model = model
        self.forward_fn = forward_fn

    def _setup_training(self) -> None:
        """Setup training components (optimizer, scheduler, criterion, etc)."""
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler(self.optimizer)

        self.weights_dir = os.path.join(self.config.log_dir, 'weights')
        os.makedirs(self.weights_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        # Training clock
        self.clock = TrainClock()

    def _get_criterion(self) -> nn.Module:
        """Initialize the loss criterion."""
        criterion_name = self.config.criterion['name']
        if criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_name == 'focal_loss':
            return FocalLoss(
                gamma=self.config.criterion['gamma'],
                reduction=self.config.criterion['reduction'])
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    def _get_optimizer(self) -> Optimizer:
        """Initialize the optimizer."""
        optimizer_config = self.config.optimizer
        if optimizer_config['name'] == 'Adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                betas=optimizer_config['betas'],
                weight_decay=optimizer_config['weight_decay']
            )
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

    def _get_scheduler(self, optimizer) -> _LRScheduler:
        """Initialize the learning rate scheduler."""
        scheduler_build = TriStageLRSchedulerLambda(
            base_lr=self.config.lr,
            total_steps=self.config.epochs,
            final_decay=self.config.scheduler['final_decay'],
            phase_ratio=self.config.scheduler['phase_ratio'])
        return scheduler_build.get_scheduler(optimizer)

    def forward(self, model, data, device):
        """Forward pass function."""
        return self.forward_fn(model, data, device)

    def train_epoch(self, dataloader):
        """Run one epoch of training."""
        loss_metric = AverageMeter('loss')

        self.model.train()
        logger.info(
            f'Training - Epoch {self.clock.epoch} started ------------------')

        predictions = np.array([])
        targets = np.array([])

        for data in tqdm(dataloader):
            # Zero to parameter gradients
            self.optimizer.zero_grad()

            # Get predections
            target = data['annotations'].to(self.device)
            model_prediction = self.forward(self.model, data, self.device)

            # Loss
            loss = self.criterion(model_prediction, target)

            # Metrics
            _, predected_calsses = torch.max(model_prediction, dim=1)
            predictions = np.append(
                predictions, predected_calsses.detach().cpu().numpy())
            targets = np.append(targets, target.detach().cpu().numpy())

            # Backward and optimizr step
            loss.backward()
            self.optimizer.step()

            # Update metrics
            loss_metric.update(loss.item(), target.size(0))

            # self.clock
            self.clock.next_step()

        logger.info(
            f'Training - Classification report, epoch {self.clock.epoch}')
        wf1, acc, bal_acc = prediction_results(predictions, targets)

        logger.info(
            f'Epoch {self.clock.epoch} training loss: {loss_metric.avg}')
        return loss_metric.avg, acc, bal_acc, wf1

    def validate_epoch(self, dataloader):
        """Run one epoch of validation."""
        loss_metric = AverageMeter('loss')

        self.model.eval()
        logger.info(
            f'Validation - Epoch {self.clock.epoch} started ------------------')

        predictions = np.array([])
        targets = np.array([])

        with torch.no_grad():
            for data in tqdm(dataloader):
                # Get predections
                target = data['annotations'].to(self.device)
                model_prediction = self.forward(
                    self.model, data, self.device)

                # Loss
                loss = self.criterion(model_prediction, target)

                # Metrics
                _, predected_calsses = torch.max(model_prediction, dim=1)
                predictions = np.append(
                    predictions, predected_calsses.cpu().numpy())
                targets = np.append(targets, target.cpu().numpy())

                # Update metrics
                loss_metric.update(loss.item(), target.size(0))

        logger.info(f'Validation - Classification report, epoch {self.clock.epoch}')
        wf1, acc, bal_acc = prediction_results(predictions, targets)

        logger.info(f'Epoch {self.clock.epoch} validation loss: {loss_metric.avg}')

        return loss_metric.avg, acc, bal_acc, wf1

    def fit(self, data, log_config=None):
        """Main training loop."""
        config = self.config
        logger.info('Start training')

        # Weights and tensorboard directories
        self._setup_training()

        # Model
        self.model = self.model.to(config.device)
        logger.info(f'Model device: {next(self.model.parameters()).device}')

        if log_config:
            formated_config = format_config(log_config, indent=4)
            self.writer.add_text('Model Config', formated_config)
            self.writer.flush()

        # NOTE: Handle continuing the training from a checkpoint

        # Dataset
        train_dataset = data.get_dataset(
            split='train',
            debug=config.debug)
        if config.oversample:
            sampler = WeightedRandomSampler(
                weights=train_dataset.weights,
                num_samples=len(train_dataset),
                replacement=True)
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                sampler=sampler)
            logger.info('Oversampling enabled')
        else:
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                shuffle=True)

        val_dataset = data.get_dataset(split='val', debug=config.debug)
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=False)

        # Track the best model
        best_loss, best_loss_epoch = np.inf, 0

        for epoch in range(config.epochs):
            logger.info(f'Epoch {epoch} started')

            train_loss, train_acc, train_bal_acc, train_f1 = self.train_epoch(
                dataloader=train_dataloader)

            val_loss, val_acc, val_bal_acc, val_wf1 = self.validate_epoch(
                dataloader=val_dataloader)

            if epoch != 0 and val_loss < best_loss:
                best_loss = val_loss
                best_loss_epoch = epoch
                self.save_checkpoint(f'{self.weights_dir}/best_loss.pth')

            # Save checkpoint each config.checkpoint_interval
            if epoch != 0 and epoch % config.checkpoint_interval == 0:
                self.save_checkpoint(f'{self.weights_dir}/checkpoint_{epoch}.pth')

            # Log learning rate
            current_lr = self.optimizer.param_groups[-1]['lr']
            logger.info(f'Learning rate at epoch {epoch}: {current_lr}')

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar(
                'Balanced Accuracy/train', train_bal_acc, epoch)
            self.writer.add_scalar('Weighted F1/train', train_f1, epoch)

            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Balanced Accuracy/val', val_bal_acc, epoch)
            self.writer.add_scalar('Weighted F1/val', val_wf1, epoch)

            self.writer.add_scalar('Learning rate', current_lr, epoch)

            self.writer.flush()

            self.scheduler.step()
            self.clock.next_epoch()

            logger.info(f'Epoch {epoch} finished')

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        logger.info('Training finished')
        logger.info(f'Best loss: {best_loss} at epoch {best_loss_epoch}')
        self.save_checkpoint(f'{self.weights_dir}/checkpoint_{self.clock.epoch}.pth')

    def save_checkpoint(self, path):
        logger.info(f'Saving checkpoint at epoch {self.clock.epoch} to {path}')
        state = {
            'clock': self.clock.state_dict(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        with open(path, 'wb') as f:
            torch.save(state, f)

# class TestBase:
#     def __init__(self, model, checkpoint_path, device, forward_fn):
#         self.model = model
#         self.device = device
#         self.forward_fn = forward_fn

#         self.load_model(checkpoint_path)

#     def load_model(self, checkpoint_path):
#         logger.info(f'Loading model from {checkpoint_path}')
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         logger.info('Clock info:')
#         logger.info(f'{json.dumps(checkpoint["clock"], indent=4)}')

#         self.model.load_state_dict(checkpoint['model'])
#         logger.info('Model loaded')

#     def test(self, data, split='test'):
#         model.eval()
#         test_dataset = data.get_dataset(split=split)
#         test_dataloader = DataLoader(
#             dataset=test_dataset,
#             batch_size=1,
#             num_workers=1,
#             shuffle=False)

#         predictions = np.array([])
#         ground_truth = np.array([])

#         with torch.no_grad():
#             for data in tqdm(test_dataloader):
#                 target = data['annotations'].to(device)

#                 model_prediction = forward_fn(model, data, device)
#                 _, predected_calsses = torch.max(model_prediction, dim=1)

#                 predictions = np.append(
#                     predictions, predected_calsses.cpu().numpy())
#                 ground_truth = np.append(ground_truth, target.cpu().numpy())

#         return ground_truth, predictions

def base_test(model, data, checkpoint_path, device, forward_fn, split='test'):
    logger.info('Start testing')

    logger.info(f'Using device: {device}')
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info('Clock info:')
    logger.info(f'{json.dumps(checkpoint["clock"], indent=4)}')

    model.load_state_dict(checkpoint['model'])
    model.eval()
    logger.info('Model loaded')

    test_dataset = data.get_dataset(split=split)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False)

    predictions = np.array([])
    ground_truth = np.array([])
    raw_predections = []

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            target = data['annotations'].to(device)
            model_prediction = forward_fn(model, data, device)
            _, predected_calsses = torch.max(model_prediction, dim=1)

            raw_predections.append(model_prediction.cpu().numpy())
            predictions = np.append(
                predictions, predected_calsses.cpu().numpy())
            ground_truth = np.append(ground_truth, target.cpu().numpy())

    return ground_truth, predictions, raw_predections

class BasePipeline:
    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def fit(self, config, model, data, log_config=None):
        trainer = BaseTrainer(
            model=model,
            config=config, 
            forward_fn=self.forward_fn)
        trainer.fit(
            data=data, 
            log_config=log_config)
    
    def test(self, model, data, checkpoint_path, device, split='test'):
        return base_test(
            model=model, 
            data=data, 
            checkpoint_path=checkpoint_path, 
            device=device,
            split=split,
            forward_fn=self.forward_fn)

