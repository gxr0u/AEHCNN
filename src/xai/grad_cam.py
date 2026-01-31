"""
Grad-CAM Implementation for EEG-Based Models

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to provide post-hoc interpretability for CNN-based EEG classification models.

The implementation is model-agnostic and supports spatio-temporal feature maps.
Visualization and result persistence are intentionally excluded as the
associated manuscript is currently under review.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for CNN-based EEG Models.

    Computes class-discriminative activation maps by leveraging gradients
    flowing into a target convolutional layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        device: torch.device,
    ):
        """
        Args:
            model (torch.nn.Module): Trained model.
            target_layer (torch.nn.Module): Convolutional layer to analyze.
            device (torch.device): Computation device.
        """
        self.model = model.to(device)
        self.target_layer = target_layer
        self.device = device

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks to capture activations and gradients.
        """

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for a given input.

        Args:
            input_tensor (torch.Tensor): Input EEG tensor of shape
                                         (1, channels, time_steps).
            target_class (int, optional): Target class index.
                                          If None, uses predicted class.

        Returns:
            torch.Tensor: Grad-CAM heatmap normalized to [0, 1].
        """
        self.model.eval()

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Compute channel-wise weights
        weights = torch.mean(self.gradients, dim=2, keepdim=True)

        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize heatmap
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze()

