"""
LIME Explainer for EEG Mind Wandering Classification

This module implements Local Interpretable Model-agnostic Explanations (LIME)
to provide instance-level interpretability for EEG-based classification models.

Visualization and persistence of explanations are intentionally excluded
as the associated manuscript is currently under review.
"""

from typing import Callable

import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer


class LimeEEGExplainer:
    """
    LIME Explainer for EEG Models.

    This class wraps LIME's tabular explainer to support EEG inputs by
    flattening spatio-temporal representations while maintaining a
    mapping back to channel-time structure.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: list,
        class_names: list,
        mode: str = "classification",
    ):
        """
        Args:
            training_data (np.ndarray): Background EEG data of shape
                                        (num_samples, num_features).
            feature_names (list): Names of flattened EEG features.
            class_names (list): Class label names.
            mode (str): Explanation mode.
        """
        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True,
        )

    def explain_instance(
        self,
        eeg_sample: torch.Tensor,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 500,
    ):
        """
        Generate a LIME explanation for a single EEG instance.

        Args:
            eeg_sample (torch.Tensor): Input EEG tensor of shape
                                       (channels, time_steps).
            predict_fn (Callable): Prediction function mapping numpy arrays
                                   to class probabilities.
            num_features (int): Number of features to include in explanation.
            num_samples (int): Number of perturbation samples.

        Returns:
            lime.explanation.Explanation: LIME explanation object.
        """
        eeg_np = eeg_sample.detach().cpu().numpy()
        eeg_flat = eeg_np.flatten()

        explanation = self.explainer.explain_instance(
            data_row=eeg_flat,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

        return explanation

