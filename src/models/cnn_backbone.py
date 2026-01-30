
"""
Hybrid CNN Backbone for EEG Feature Extraction

This module implements a spatio-temporal convolutional backbone designed
to extract meaningful representations from multi-channel EEG signals.

The backbone is intentionally modular to support architectural ablation
and extension in research settings.
"""

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    Hybrid CNN Backbone.

    This backbone extracts hierarchical spatio-temporal features from EEG
    signals using a combination of temporal and spatial convolutions.

    Expected input shape:
        (batch_size, num_channels, time_steps)

    Output shape:
        (batch_size, num_features, feature_dim)
    """

    def __init__(
        self,
        num_channels: int,
        feature_dim: int = 128,
        temporal_kernel_sizes=(7, 5, 3),
        spatial_kernel_size: int = 1,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_channels (int): Number of EEG channels.
            feature_dim (int): Dimensionality of the output feature vectors.
            temporal_kernel_sizes (tuple): Kernel sizes for temporal convolutions.
            spatial_kernel_size (int): Kernel size for spatial convolutions.
            dropout (float): Dropout probability.
        """
        super(CNNBackbone, self).__init__()

        # Temporal convolutional blocks
        temporal_layers = []
        in_channels = num_channels

        for kernel_size in temporal_kernel_sizes:
            temporal_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=feature_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_channels = feature_dim

        self.temporal_conv = nn.Sequential(*temporal_layers)

        # Spatial convolution to model inter-channel dependencies
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=spatial_kernel_size,
            ),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN backbone.

        Args:
            x (torch.Tensor): Input EEG tensor of shape
                              (batch_size, num_channels, time_steps)

        Returns:
            torch.Tensor: Feature tensor of shape
                          (batch_size, num_features, feature_dim)
        """
        # Temporal feature extraction
        x = self.temporal_conv(x)
        # Shape: (batch_size, feature_dim, time_steps)

        # Spatial feature extraction
        x = self.spatial_conv(x)
        # Shape: (batch_size, feature_dim, time_steps)

        # Transpose to (batch_size, num_features, feature_dim)
        x = x.permute(0, 2, 1)

        return x
