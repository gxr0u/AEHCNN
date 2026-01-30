"""
Hybrid CNN Model with Attention for EEG Mind Wandering Classification

This module assembles the full Attention-Enhanced Hybrid CNN architecture by
integrating:
1. CNN-based spatio-temporal feature extraction
2. Attention-based feature reweighting
3. A classification head for cognitive state prediction
"""

import torch
import torch.nn as nn

from src.models.attention import AttentionLayer
from src.models.cnn_backbone import CNNBackbone


class HybridAttentionCNN(nn.Module):
    """
    Attention-Enhanced Hybrid CNN Model.

    This model combines a CNN backbone for feature extraction with an
    attention mechanism for adaptive feature weighting, followed by a
    classification head.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        feature_dim: int = 128,
        use_attention: bool = True,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_channels (int): Number of EEG channels.
            num_classes (int): Number of output classes.
            feature_dim (int): Dimensionality of extracted feature vectors.
            use_attention (bool): Whether to enable attention mechanism.
            dropout (float): Dropout probability for regularization.
        """
        super(HybridAttentionCNN, self).__init__()

        self.use_attention = use_attention

        # CNN backbone for spatio-temporal feature extraction
        self.backbone = CNNBackbone(
            num_channels=num_channels,
            feature_dim=feature_dim,
        )

        # Attention module
        if self.use_attention:
            self.attention = AttentionLayer(feature_dim=feature_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input EEG tensor of shape
                              (batch_size, num_channels, time_steps)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract spatio-temporal features
        features = self.backbone(x)
        # Expected shape: (batch_size, num_features, feature_dim)

        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        else:
            # Global average pooling as a baseline (no attention)
            features = features.mean(dim=1)

        # Classification
        logits = self.classifier(features)

        return logits

