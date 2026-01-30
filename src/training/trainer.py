"""
Training Utilities for EEG Mind Wandering Classification

This module implements a generic training loop for deep learning models,
designed to support reproducible research experiments and ablation studies.

Numerical results and checkpoints are intentionally not persisted publicly
as the associated manuscript is currently under review.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Generic Trainer Class.

    Handles model training and validation while maintaining clean separation
    between model definition, optimization, and evaluation logic.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            model (nn.Module): Model to be trained.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            criterion (nn.Module): Loss function.
            device (torch.device): Training device (CPU or GPU).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): Training data loader.

        Returns:
            Dict[str, float]: Dictionary containing training metrics.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)

        return {
            "train_loss": avg_loss
        }

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model for one epoch.

        Args:
            dataloader (DataLoader): Validation data loader.

        Returns:
            Dict[str, float]: Dictionary containing validation metrics.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)

        return {
            "val_loss": avg_loss
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Tuple[list, list]:
        """
        Full training loop across multiple epochs.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            num_epochs (int): Number of training epochs.

        Returns:
            Tuple[list, list]: Training and validation history.
        """
        train_history = []
        val_history = []

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            train_history.append(train_metrics)
            val_history.append(val_metrics)

            # Minimal console logging (no result leakage)
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )

        return train_history, val_history

