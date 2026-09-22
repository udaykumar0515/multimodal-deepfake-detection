import torch
import torch.nn as nn
from typing import Optional, Dict

from .visual_encoder import VisualEncoder
from .audio_encoder import AudioEncoder

class MultiHeadDeepfakeModel(nn.Module):
    """
    MultiHeadDeepfakeModel
    ======================
    A flexible multi-branch architecture supporting independent image, audio, and fused video+audio predictions.
    
    Heads:
    - image_head: predicts deepfake probability from image features (or independent video frames).
    - audio_head: predicts deepfake probability from audio log-mel features.
    - fusion_head: predicts deepfake probability from combined video and audio features.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder(pretrained=pretrained)
        self.audio_encoder = AudioEncoder(pretrained=pretrained)

        # Modality-specific heads
        self.image_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 1)
        )
        
        self.audio_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 1)
        )

        # Multimodal fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )

    def forward(
        self, 
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Flexible forward pass.
        
        Args:
            image: (B, 3, 224, 224)
            video: (B, T, 3, 224, 224)
            audio: (B, 3, 224, 224)
            return_all: If True, computes and returns all applicable heads (useful for multi-task training).
            
        Returns:
            A dictionary containing raw logits for 'image', 'audio', and/or 'fusion'.
        """
        outputs = {}

        # 1. Standalone Image Inference
        if image is not None and video is None and audio is None:
            image_feat = self.visual_encoder(image)
            outputs['image'] = self.image_head(image_feat)
            return outputs

        # 2. Standalone Audio Inference
        if audio is not None and video is None and image is None:
            audio_feat = self.audio_encoder(audio)
            outputs['audio'] = self.audio_head(audio_feat)
            return outputs

        # 3. Multimodal Video + Audio Inference (and optionally multi-task outputs)
        if video is not None and audio is not None:
            if return_all:
                video_feat, frame_feats = self.visual_encoder(video, return_unpooled=True)
                audio_feat = self.audio_encoder(audio)
                
                fused = torch.cat([video_feat, audio_feat], dim=1)
                
                outputs['fusion'] = self.fusion_head(fused)
                outputs['audio'] = self.audio_head(audio_feat)
                outputs['image'] = self.image_head(frame_feats) # Evaluates all B*T frames independently
            else:
                video_feat = self.visual_encoder(video, return_unpooled=False)
                audio_feat = self.audio_encoder(audio)
                
                fused = torch.cat([video_feat, audio_feat], dim=1)
                outputs['fusion'] = self.fusion_head(fused)
                
            return outputs

        raise ValueError("Invalid combination of inputs. Provide either (image), (audio), or (video, audio).")
