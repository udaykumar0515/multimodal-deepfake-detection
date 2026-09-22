"""
MultimodalDeepfakeModel
=======================
Combines VideoEncoder + AudioEncoder with a fusion classifier head.

Pipeline:
    video (B, 16, 3, 224, 224) ─► VideoEncoder ─► (B, 1280) ─┐
                                                               Concat → (B, 2560)
    audio (B, 3, 224, 224)     ─► AudioEncoder ─► (B, 1280) ─┘
                                                               ↓
                                                    Linear(2560 → 512)
                                                               ↓
                                                             ReLU
                                                               ↓
                                                        Dropout(0.3)
                                                               ↓
                                                    Linear(512 → 1)
                                                               ↓
                                                     raw logit (B, 1)

No sigmoid is applied inside the model; BCEWithLogitsLoss / Focal Loss
applied externally during training will supply it.

Grad-CAM access:
    model.video_encoder.grad_cam_layer  – final conv block of video EfficientNet
    model.audio_encoder.grad_cam_layer  – final conv block of audio EfficientNet
"""

import torch
import torch.nn as nn

from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder


class MultimodalDeepfakeModel(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.video_encoder = VideoEncoder(pretrained=pretrained)
        self.audio_encoder = AudioEncoder(pretrained=pretrained)

        # Fusion classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        video: torch.Tensor,   # (B, 16, 3, 224, 224)
        audio: torch.Tensor,   # (B, 3, 224, 224)
    ) -> torch.Tensor:         # (B, 1)  raw logit

        video_feat = self.video_encoder(video)   # (B, 1280)
        audio_feat = self.audio_encoder(audio)   # (B, 1280)

        fused = torch.cat([video_feat, audio_feat], dim=1)  # (B, 2560)
        logit = self.classifier(fused)                      # (B, 1)
        return logit
