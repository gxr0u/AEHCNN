"""
Evaluation Script for Attention-Enhanced Hybrid CNN

This script serves as the main entry point for evaluating a trained EEG
Mind Wandering classification model on held-out data.

Note:
- Dataset loading and trained model weights are intentionally excluded.
- Numerical results and visualizations are not persisted publicly as the
  associated manuscript is currently under review.
"""

import torch
from torch.utils.data import DataLoader

from src.config import (
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    initialize_experiment,
)
from src.models.hybrid_model import HybridAttentionCNN
from src.evaluation.evaluate import Evaluator


def get_evaluation_dataloader(batch_size: int):
    """
    Placeholder function for evaluation data loading.

    Replace this with the actual EEG dataset and preprocessing pipeline
    during controlled experiments.

    Args:
        batch_size (int): Batch size for evaluation.

    Returns:
        DataLoader: Evaluation data loader.
    """
    raise NotImplementedError(
        "Evaluation data loading is not included as the paper is under review."
    )


def load_trained_model(model: torch.nn.Module):
    """
    Placeholder for loading trained model weights.

    Model checkpoints are not publicly released while the manuscript
    is under review.

    Args:
        model (torch.nn.Module): Model architecture instance.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    raise NotImplementedError(
        "Trained model weights are not publicly available."
    )


def main():
    """
    Main evaluation routine.
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
    # Load trained weights (placeholder)
    # -----------------------------
    model = load_trained_model(model)

    # -----------------------------
    # Evaluation data loader
    # -----------------------------
    eval_loader = get_evaluation_dataloader(train_cfg.batch_size)

    # -----------------------------
    # Evaluator
    # -----------------------------
    evaluator = Evaluator(
        model=model,
        device=device,
    )

    # -----------------------------
    # Run evaluation
    # -----------------------------
    metrics = evaluator.evaluate(eval_loader)

    # -----------------------------
    # Console output (minimal, review-safe)
    # -----------------------------
    print("Evaluation completed.")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
