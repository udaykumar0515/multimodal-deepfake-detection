"""
Binary Focal Loss
=================
Numerically stable binary focal loss operating on raw logits.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Focal Loss down-weights easy negatives and focuses training on
hard examples — an important complement to the WeightedRandomSampler
already handling class-imbalance at the sampling level.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

For binary case with raw logits:
    p = sigmoid(logit)
    FL = -(1 - p)^gamma * log(p)          for positive labels
    FL = -p^gamma * log(1 - p)             for negative labels

Uses torch.nn.functional.binary_cross_entropy_with_logits internally
for numerical stability (log-sum-exp trick avoids overflow).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for raw logit inputs.

    Args:
        gamma (float): Focusing parameter. 0.0 reduces to standard BCE.
        reduction (str): 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1) or (B,) — raw unnormalised model output
            targets: (B,)           — float32 binary labels {0.0, 1.0}

        Returns:
            Scalar loss (if reduction='mean') or per-element tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Numerically stable BCE per element
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t: probability for the true class
        # = sigmoid(logit)  when target==1
        # = 1 - sigmoid(logit)  when target==0
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1.0, p, 1.0 - p)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
