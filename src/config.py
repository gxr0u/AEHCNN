"""
Global Configuration and Reproducibility Utilities

This module centralizes all experiment-level configurations including
model hyperparameters, training settings, and reproducibility controls.

Having a single configuration source ensures consistent experiments
and facilitates ablation studies.
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelConfig:
    """
    Configuration for model architecture.
    """
    num_channels: int
    num_classes: int = 2
    feature_dim: int = 128
    use_attention: bool = True
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    """
    Configuration for training procedure.
    """
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "step"
    step_size: int = 20
    gamma: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """
    Configuration for experiment management.
    """
    experiment_name: str = "attention_hybrid_cnn_eeg"
    seed: int = 42
    log_dir: str = "experiments/runs"
    save_checkpoints: bool = False
    notes: str = "Paper under review; results withheld"


def initialize_experiment(config: ExperimentConfig) -> None:
    """
    Initialize experiment settings including reproducibility and logging.

    Args:
        config (ExperimentConfig): Experiment configuration.
    """
    set_global_seed(config.seed)

    os.makedirs(config.log_dir, exist_ok=True)

