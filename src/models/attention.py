
"""
Attention Module for EEG Feature Reweighting

This module implements a lightweight attention mechanism designed to
adaptively emphasize salient spatio-temporal EEG features learned by
the hybrid CNN backbone.

The attention mechanism is trained end-to-end jointly with the model
and serves both performance enhancement and interpretability purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Generic Attention Layer for Feature Reweighting.

    This module computes attention weights over feature representations
    and produces an attention-weighted feature vector.

    Expected input shape:
        (batch_size, num_features, feature_dim)

    Output shape:
        (batch_size, feature_dim)
    """

    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim (int): Dimensionality of the input feature vectors.
        """
        super(AttentionLayer, self).__init__()

        self.feature_dim = feature_dim

        # Learnable attention scoring function
        self.attention_fc = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the attention layer.

        Args:
            x (torch.Tensor): Input feature tensor of shape
                              (batch_size, num_features, feature_dim)

        Returns:
            torch.Tensor: Attention-weighted feature representation of shape
                          (batch_size, feature_dim)
        """
        # Compute unnormalized attention scores
        # Shape: (batch_size, num_features, 1)
        scores = self.attention_fc(x)

        # Normalize scores to obtain attention weights
        # Shape: (batch_size, num_features, 1)
        attention_weights = F.softmax(scores, dim=1)

        # Apply attention weights
        # Shape: (batch_size, feature_dim)
        weighted_features = torch.sum(attention_weights * x, dim=1)

        return weighted_features


class ChannelWiseAttention(AttentionLayer):
    """
    Channel-wise Attention Mechanism.

    This specialization applies attention across EEG channels, allowing
    the model to emphasize spatially relevant brain regions.
    """

    def __init__(self, feature_dim: int):
        super(ChannelWiseAttention, self).__init__(feature_dim)


class TemporalAttention(AttentionLayer):
    """
    Temporal Attention Mechanism.

    This specialization applies attention across temporal feature locations,
    enabling the model to focus on time segments associated with mind wandering.
    """

    def __init__(self, feature_dim: int):
        super(TemporalAttention, self).__init__(feature_dim)
