import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Tuple, Union


class VisualEncoder(nn.Module):
    """
    VisualEncoder
    =============
    Encodes images or videos using EfficientNet-B0.

    Input shapes supported:
      - Image: (B, 3, 224, 224)
      - Video: (B, T, 3, 224, 224)

    For video inputs, it can return either the mean-pooled sequence representation (B, 1280),
    or both the sequence representation and the unpooled frame features (B*T, 1280) for frame-level analysis.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Keep the feature extractor and adaptive pooling
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Expose last conv block for Grad-CAM
        self.grad_cam_layer: nn.Module = self.features[-1]

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a single image tensor.
        Args:
            x: (B, 3, 224, 224)
        Returns:
            (B, 1280)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x

    def forward_video(self, x: torch.Tensor, return_unpooled: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a sequence of video frames.
        Args:
            x: (B, T, 3, 224, 224)
            return_unpooled: If True, also return the (B*T, 1280) tensor.
        Returns:
            sequence_rep: (B, 1280)
            [optional] frame_rep: (B*T, 1280)
        """
        B, T, C, H, W = x.shape

        # Fold time into batch dimension
        x_flat = x.view(B * T, C, H, W)
        
        # Forward pass through backbone
        feat = self.features(x_flat)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)  # (B*T, 1280)
        
        # Unfold time dimension and pool
        feat_seq = feat.view(B, T, -1)  # (B, T, 1280)
        seq_rep = feat_seq.mean(dim=1)  # (B, 1280)

        if return_unpooled:
            return seq_rep, feat
        return seq_rep

    def forward(self, x: torch.Tensor, return_unpooled: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Auto-detects whether the input is an image or a video based on dimensionality.
        - 4D tensor -> treated as (B, 3, 224, 224) Image
        - 5D tensor -> treated as (B, T, 3, 224, 224) Video
        """
        if x.dim() == 4:
            return self.forward_image(x)
        elif x.dim() == 5:
            return self.forward_video(x, return_unpooled)
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got {x.dim()}D.")
