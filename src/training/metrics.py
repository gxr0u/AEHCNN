"""
Evaluation Metrics for EEG Mind Wandering Classification

This module defines commonly used classification metrics for evaluating
EEG-based cognitive state classification models.

Metric computation is intentionally decoupled from result reporting to
support controlled experimentation and review-safe code release.
"""

from typing import Dict

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    return tensor.detach().cpu().numpy()


def classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_prob: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Compute classification metrics for binary classification.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted class labels.
        y_prob (torch.Tensor, optional): Predicted probabilities for the
                                         positive class.

    Returns:
        Dict[str, float]: Dictionary of computed metrics.
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
        "f1_score": f1_score(y_true_np, y_pred_np, zero_division=0),
    }

    if y_prob is not None:
        y_prob_np = _to_numpy(y_prob)
        metrics["roc_auc"] = roc_auc_score(y_true_np, y_prob_np)

    return metrics


def aggregate_epoch_metrics(batch_metrics: list) -> Dict[str, float]:
    """
    Aggregate batch-level metrics into epoch-level metrics.

    Args:
        batch_metrics (list): List of metric dictionaries from individual batches.

    Returns:
        Dict[str, float]: Averaged epoch-level metrics.
    """
    if not batch_metrics:
        return {}

    aggregated = {}
    for key in batch_metrics[0].keys():
        aggregated[key] = float(
            np.mean([m[key] for m in batch_metrics if key in m])
        )

    return aggregated

