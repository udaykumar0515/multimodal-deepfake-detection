"""
VideoEncoder
============
Encodes a batch of 16-frame videos through EfficientNet-B0.

Input  : (B, 16, 3, 224, 224)  – uint8-originated, already normalized float32
Output : (B, 1280)              – mean-pooled frame features

Design note (Grad-CAM):
    The final convolutional block of EfficientNet-B0 is exposed via the
    `grad_cam_layer` attribute so that a later Grad-CAM pass can register
    hooks on it without accessing private torchvision internals.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class VideoEncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Keep the convolutional feature extractor; discard the classifier head.
        # EfficientNet-B0 topology:
        #   backbone.features  → (B, 1280, 7, 7)  after global pooling → (B, 1280)
        self.features = backbone.features          # Conv blocks 0-8
        self.avgpool  = backbone.avgpool           # AdaptiveAvgPool2d(1)

        # Expose the last convolutional block for future Grad-CAM hook attachment.
        # backbone.features[-1] is Conv2dNormActivation (block 8, the "head" conv).
        self.grad_cam_layer: nn.Module = self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, 3, 224, 224)
        Returns:
            (B, 1280)
        """
        B, T, C, H, W = x.shape          # T = 16 frames

        # Merge batch and time dimensions so EfficientNet processes every frame.
        x = x.view(B * T, C, H, W)       # (B*16, 3, 224, 224)
        x = self.features(x)             # (B*16, 1280, 7, 7)
        x = self.avgpool(x)              # (B*16, 1280, 1, 1)
        x = x.flatten(1)                 # (B*16, 1280)

        # Restore temporal dimension and mean-pool over the 16 frames.
        x = x.view(B, T, -1)            # (B, 16, 1280)
        x = x.mean(dim=1)               # (B, 1280)
        return x
