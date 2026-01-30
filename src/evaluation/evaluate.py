"""
Model Evaluation Utilities for EEG Mind Wandering Classification

This module defines the evaluation pipeline for trained models, including
metric computation and prediction aggregation.

Numerical results and visual outputs are intentionally not persisted
publicly as the associated manuscript is currently under review.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import classification_metrics


class Evaluator:
    """
    Generic Evaluator Class.

    Handles model evaluation on held-out datasets while maintaining
    separation between metric computation and result reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            model (nn.Module): Trained model to be evaluated.
            device (torch.device): Evaluation device.
        """
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a given dataset.

        Args:
            dataloader (DataLoader): Evaluation data loader.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        self.model.eval()

        all_targets: List[torch.Tensor] = []
        all_preds: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_targets.append(targets)
                all_preds.append(preds)
                all_probs.append(probs[:, 1])

        y_true = torch.cat(all_targets)
        y_pred = torch.cat(all_preds)
        y_prob = torch.cat(all_probs)

        metrics = classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return metrics

