import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for raw logit inputs.
    Inherited exactly from V1 for proven numerical stability.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1.0, p, 1.0 - p)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class MultiTaskFocalLoss(nn.Module):
    """
    Multi-Task wrapper for the V2 Architecture.
    Calculates independent focal losses for image, audio, and fusion heads.
    Expands video_label to match the 16 independent frame predictions for the image head.
    """
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.focal_loss = BinaryFocalLoss(gamma=gamma, reduction="mean")
        
    def forward(self, preds: dict, video_label: torch.Tensor, audio_label: torch.Tensor, overall_label: torch.Tensor) -> dict:
        """
        Args:
            preds: dict containing 'image', 'audio', 'fusion' raw logits.
            video_label: (B, 1)
            audio_label: (B, 1)
            overall_label: (B, 1)
            
        Returns:
            dict containing: total_loss, image_loss, audio_loss, fusion_loss
        """
        # Ensure correct shapes
        video_label = video_label.view(-1, 1)
        audio_label = audio_label.view(-1, 1)
        overall_label = overall_label.view(-1, 1)
        
        # 1. Fusion Loss
        fusion_loss = self.focal_loss(preds['fusion'], overall_label)
        
        # 2. Audio Loss
        audio_loss = self.focal_loss(preds['audio'], audio_label)
        
        # 3. Image Loss
        # Expand (B, 1) video_label to (B*16, 1) mapping exactly to the 16 frame features in visual_encoder
        B_times_16 = preds['image'].size(0)
        frames_per_video = B_times_16 // video_label.size(0)
        
        # repeat_interleave ensures:
        # [vid1, vid2] -> [vid1, vid1, ..., vid2, vid2, ...]
        expanded_video_label = video_label.repeat_interleave(frames_per_video, dim=0)
        
        image_loss = self.focal_loss(preds['image'], expanded_video_label)
        
        # Equal unweighted summation
        total_loss = image_loss + audio_loss + fusion_loss
        
        return {
            'total': total_loss,
            'image': image_loss,
            'audio': audio_loss,
            'fusion': fusion_loss
        }
