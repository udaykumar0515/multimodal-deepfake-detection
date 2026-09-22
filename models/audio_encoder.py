import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AudioEncoder(nn.Module):
    """
    AudioEncoder
    ============
    Encodes a log-mel spectrogram through EfficientNet-B0.

    Input  : (B, 3, 224, 224)  – float32 spectrogram (3-channel duplicate)
    Output : (B, 1280)
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Expose last conv block for Grad-CAM
        self.grad_cam_layer: nn.Module = self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            (B, 1280)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x
