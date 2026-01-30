"""
Training Script for Attention-Enhanced Hybrid CNN

This script serves as the main entry point for training the proposed
EEG Mind Wandering classification model.

Note:
- Raw data loading is intentionally abstracted.
- Model checkpoints and numerical results are not persisted publicly
  as the associated manuscript is currently under review.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    initialize_experiment,
)
from src.models.hybrid_model import HybridAttentionCNN
from src.training.trainer import Trainer


def get_dataloaders(batch_size: int):
    """
    Placeholder function for data loading.

    Replace this with the actual EEG dataset and preprocessing pipeline
    when running controlled experiments.

    Args:
        batch_size (int): Batch size for training.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    raise NotImplementedError(
        "EEG data loading is not included as the paper is under review."
    )


def main():
    """
    Main training routine.
    """
    # -----------------------------
    # Experiment initialization
    # -----------------------------
    exp_cfg = ExperimentConfig()
    initialize_experiment(exp_cfg)

    # -----------------------------
    # Configuration setup
    # -----------------------------
    model_cfg = ModelConfig(num_channels=64)
    train_cfg = TrainingConfig()

    device = torch.device(train_cfg.device)

    # -----------------------------
    # Model initialization
    # -----------------------------
    model = HybridAttentionCNN(
        num_channels=model_cfg.num_channels,
        num_classes=model_cfg.num_classes,
        feature_dim=model_cfg.feature_dim,
        use_attention=model_cfg.use_attention,
        dropout=model_cfg.dropout,
    ).to(device)

    # -----------------------------
    # Optimization setup
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # -----------------------------
    # Data loaders (placeholder)
    # -----------------------------
    train_loader, val_loader = get_dataloaders(train_cfg.batch_size)

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # -----------------------------
    # Training loop
    # -----------------------------
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_cfg.num_epochs,
    )


if __name__ == "__main__":
    main()
