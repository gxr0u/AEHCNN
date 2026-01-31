"""
SHAP Explainer for EEG Mind Wandering Classification

This module implements SHapley Additive exPlanations (SHAP) to provide
both global and local interpretability for EEG-based classification models.

SHAP values and visualizations are intentionally not persisted publicly
as the associated manuscript is currently under review.
"""

from typing import Callable

import numpy as np
import torch
import shap


class SHAPEEGExplainer:
    """
    SHAP Explainer for EEG Models.

    This class supports feature attribution for EEG inputs by operating
    on flattened spatio-temporal representations while retaining
    compatibility with deep learning models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        device: torch.device,
    ):
        """
        Args:
            model (torch.nn.Module): Trained EEG classification model.
            background_data (np.ndarray): Background dataset used for SHAP
                                           value estimation.
            device (torch.device): Computation device.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.explainer = shap.KernelExplainer(
            self._predict,
            background_data,
        )

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prediction function compatible with SHAP.

        Args:
            x (np.ndarray): Flattened EEG input array.

        Returns:
            np.ndarray: Model prediction probabilities.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Infer channel and time dimensions dynamically
        batch_size = x_tensor.shape[0]
        feature_dim = x_tensor.shape[1]

        # Reshape must match training pipeline (handled externally)
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def explain(
        self,
        eeg_sample: np.ndarray,
        nsamples: int = 100,
    ):
        """
        Generate SHAP values for a given EEG sample.

        Args:
            eeg_sample (np.ndarray): Flattened EEG sample.
            nsamples (int): Number of samples for SHAP approximation.

        Returns:
            np.ndarray: SHAP values.
        """
        shap_values = self.explainer.shap_values(
            eeg_sample,
            nsamples=nsamples,
        )

        return shap_values

