import math
import logging
import numpy as np
import torch.optim as optim
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, balanced_accuracy_score

logger = logging.getLogger(__name__)

class TrainClock:
    # Credit: github.com/henryxrl
    """ Clock object to track epoch and step during training """

    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def next_step(self):
        self.minibatch += 1
        self.step += 1

    def next_epoch(self):
        self.epoch += 1
        self.minibatch = 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def load_state_dict(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']

class AverageMeter:
    # Credit: github.com/henryxrl
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def prediction_results(predictions, annotations):
    """
    This function takes predictions and annotations (true labels), prints the confusion matrix and classification report, 
    and returns the weighted F1 score, accuracy, and balanced accuracy.

    Parameters:
        predictions (array-like): Predicted labels.
        annotations (array-like): True labels.

    Returns:
        f1 (float): F1 score.
        accuracy (float): Accuracy score.
        balanced_accuracy (float): Balanced accuracy score.
    """
    
    # Classification Report
    report = classification_report(annotations, predictions, zero_division=0)
    logger.info(f"\n{report}")
    
    # F1 Score
    f1 = f1_score(annotations, predictions, average='weighted', zero_division=0)
    
    # Accuracy Score
    accuracy = accuracy_score(annotations, predictions)
    
    # Balanced Accuracy Score
    balanced_accuracy = balanced_accuracy_score(annotations, predictions)
    
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    
    return f1, accuracy, balanced_accuracy


class TriStageLRSchedulerLambda:
    # heavily inspired by: https://github.com/facebookresearch/fairseq.git
    def __init__(
            self, 
            base_lr, 
            total_steps, 
            phase_ratio=None, 
            init_lr_scale=0.01, 
            final_lr_scale=0.01, 
            final_decay='exp'
        ):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        if final_decay not in ('exp', 'linear', 'cosine'):
            raise ValueError('final_decay must be one of exp, linear, cosine')
        self.final_decay = final_decay

        # Handling phase_ratio
        if phase_ratio is None:
            phase_ratio = [0.1, 0.4, 0.5]
        if sum(phase_ratio) != 1:
            raise ValueError("phase ratios must add up to 1")
        self.phase_ratio = phase_ratio

        self.peak_lr = self.base_lr
        self.init_lr = self.init_lr_scale * self.base_lr
        self.final_lr = self.final_lr_scale * self.base_lr

        self.warmup_steps = int(self.total_steps * self.phase_ratio[0])
        self.hold_steps = int(self.total_steps * self.phase_ratio[1])
        self.decay_steps = int(self.total_steps * self.phase_ratio[2])

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )

    def decay_factor(self, steps_in_stage):
        if self.final_decay == 'exp':
            decay_rate = -math.log(self.final_lr_scale) / self.decay_steps
            factor = math.exp(-decay_rate * steps_in_stage)
            return factor
        elif self.final_decay == 'linear':
            decay_rate = (
                (self.final_lr - self.peak_lr) / self.decay_steps
                if self.decay_steps != 0
                else 0
            )
            factor = 1 + (decay_rate * steps_in_stage) / self.base_lr
            return factor
        elif self.final_decay == 'cosine':
            decay_rate = (
                steps_in_stage / self.decay_steps
                if self.decay_steps != 0
                else 0
            )
            factor = self.final_lr_scale + \
                (1 - self.final_lr_scale) * 0.5 * (1. + math.cos(math.pi * decay_rate))
            return factor

    def decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def lr_lambda(self, step):
        """Update the learning rate after each update."""
        stage, steps_in_stage = self.decide_stage(step)
        if stage == 0:
            factor = self.init_lr_scale + (self.warmup_rate * steps_in_stage) / self.base_lr
        elif stage == 1:
            factor = 1
        elif stage == 2:
            factor = self.decay_factor(steps_in_stage)
        elif stage == 3:
            factor = self.final_lr_scale

        return factor

    def get_scheduler(self, optimizer):
        """Return the LambdaLR scheduler."""
        return optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

class TriStageLRScheduler_Over(LRScheduler):
    def __init__(self, optimizer, total_steps, phase_ratio=None,
                 init_lr_scale=0.01, final_lr_scale=0.01, final_decay='exp', last_epoch=-1, verbose=False):
        """
        Custom learning rate scheduler with warm-up, constant, and decay phases.

        Parameters:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs to be scheduled.
            total_steps (int): Total number of training steps.
            phase_ratio (list): List with 3 ratios for the warm-up, hold, and decay phases (default: [0.1, 0.4, 0.5]).
            init_lr_scale (float): Scale factor for the initial learning rate (during warm-up).
            final_lr_scale (float): Scale factor for the final learning rate (after decay).
            final_decay (str): Type of decay: 'exp', 'linear', or 'cosine' (default: 'exp').
            last_epoch (int): The index of the last epoch when resuming training. Default: -1.
            verbose (bool): If True, prints learning rate updates.
        """
        if phase_ratio is None:
            phase_ratio = [0.1, 0.4, 0.5]

        assert sum(phase_ratio) == 1, "phase_ratio must sum to 1"

        self.total_steps = total_steps
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale
        self.final_decay = final_decay
        self.phase_ratio = phase_ratio

        self.warmup_steps = int(total_steps * phase_ratio[0])
        self.hold_steps = int(total_steps * phase_ratio[1])
        self.decay_steps = int(total_steps * phase_ratio[2])

        # Compute the initial and final learning rates
        self.peak_lr = optimizer.param_groups[0]['lr']
        self.init_lr = init_lr_scale * self.peak_lr
        self.final_lr = final_lr_scale * self.peak_lr

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute the new learning rate based on the current epoch."""
        step = self.last_epoch

        # Choose the appropriate stage
        if step < self.warmup_steps:
            # Warm-up stage
            factor = self._warmup_factor(step)
        elif step < self.warmup_steps + self.hold_steps:
            # Hold stage
            factor = 1
        elif step < self.total_steps:
            # Decay stage
            steps_in_decay = step - (self.warmup_steps + self.hold_steps)
            factor = self._decay_factor(steps_in_decay)
        else:
            # After decay, constant final lr
            factor = self.final_lr_scale

        # Update the learning rate for all parameter groups
        return [self.peak_lr * factor]

    def _warmup_factor(self, step):
        """Compute the warm-up factor."""
        init_lr = self.init_lr
        peak_lr = self.peak_lr
        warmup_steps = self.warmup_steps
        return (init_lr + (peak_lr - init_lr) * (step / warmup_steps)) / peak_lr

    def _decay_factor(self, steps_in_stage):
        """Compute the decay factor based on the selected decay method."""
        if self.final_decay == 'exp':
            return math.exp(math.log(self.final_lr_scale) * steps_in_stage / self.decay_steps)
        elif self.final_decay == 'linear':
            peak_lr = self.peak_lr
            final_lr = self.final_lr
            decay_steps = self.decay_steps
            return (final_lr + (peak_lr - final_lr) * (1 - steps_in_stage / decay_steps)) / peak_lr
        elif self.final_decay == 'cosine':
            final_lr_scale = self.final_lr_scale
            decay_steps = self.decay_steps
            return final_lr_scale + (1 - final_lr_scale) * 0.5 * (1 + math.cos(math.pi * steps_in_stage / decay_steps))
        else:
            raise NotImplementedError(f"Decay method '{self.final_decay}' is not implemented")

# Logging configuration into TensorBoard for convenience
def format_config(config, indent=0):
    """
    Recursively formats the dataclass config into a human-readable string.
    """
    config_str = ""
    for key, value in config.items():
        if isinstance(value, dict):
            config_str += " " * indent + f"{key}:\n" + format_config(value, indent + 4)
        else:
            config_str += " " * indent + f"{key}: {value}\n"
    return config_str


class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, gamma=2., reduction='none'):
        super().__init__()
        self.weight = torch.tensor(weight) if weight is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predection, target):
        # NOTE: Temporal fix for the device issue
        if self.weight is not None:
            self.weight = self.weight.to(predection.device)
        log_prob = F.log_softmax(predection, dim=-1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target, 
            weight=self.weight,
            reduction = self.reduction
        )
        return loss