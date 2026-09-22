import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm

from .losses import MultiTaskFocalLoss

class Trainer:
    """
    V2 Multi-Task Trainer for Deepfake Detection.
    Jointly optimizes Image, Audio, and Fusion heads via equal unweighted summation.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
        gradient_clip: float = 1.0,
        device: torch.device = torch.device('cpu'),
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        
        self.criterion = MultiTaskFocalLoss(gamma=focal_gamma).to(device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )
        self.scaler = GradScaler()
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.latest_checkpoint_path = self.checkpoint_dir / "latest_model.pt"
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_loss, total_img, total_aud, total_fus = 0.0, 0.0, 0.0, 0.0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            video, audio, v_label, a_label, o_label = [x.to(self.device) for x in batch]
            
            self.optimizer.zero_grad()
            
            with autocast():
                preds = self.model(video=video, audio=audio, return_all=True)
                losses = self.criterion(preds, v_label, a_label, o_label)
                
            self.scaler.scale(losses['total']).backward()
            
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += losses['total'].item()
            total_img += losses['image'].item()
            total_aud += losses['audio'].item()
            total_fus += losses['fusion'].item()
            
            pbar.set_postfix({"Loss": losses['total'].item()})
            
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'image_loss': total_img / num_batches,
            'audio_loss': total_aud / num_batches,
            'fusion_loss': total_fus / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        self.model.eval()
        
        total_loss, total_img, total_aud, total_fus = 0.0, 0.0, 0.0, 0.0
        all_preds_fus, all_targets_fus = [], []
        all_preds_aud, all_targets_aud = [], []
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        for batch in pbar:
            video, audio, v_label, a_label, o_label = [x.to(self.device) for x in batch]
            
            with autocast():
                preds = self.model(video=video, audio=audio, return_all=True)
                losses = self.criterion(preds, v_label, a_label, o_label)
                
            total_loss += losses['total'].item()
            total_img += losses['image'].item()
            total_aud += losses['audio'].item()
            total_fus += losses['fusion'].item()
            
            # Binary classification metrics for fusion & audio
            fusion_preds = torch.sigmoid(preds['fusion']) > 0.5
            audio_preds = torch.sigmoid(preds['audio']) > 0.5
            
            all_preds_fus.extend(fusion_preds.cpu().numpy())
            all_targets_fus.extend(o_label.cpu().numpy())
            
            all_preds_aud.extend(audio_preds.cpu().numpy())
            all_targets_aud.extend(a_label.cpu().numpy())
            
        num_batches = len(self.val_loader)
        
        fusion_acc = accuracy_score(all_targets_fus, all_preds_fus)
        fusion_f1 = f1_score(all_targets_fus, all_preds_fus, zero_division=0)
        audio_acc = accuracy_score(all_targets_aud, all_preds_aud)
        audio_f1 = f1_score(all_targets_aud, all_preds_aud, zero_division=0)
        
        return {
            'loss': total_loss / num_batches,
            'image_loss': total_img / num_batches,
            'audio_loss': total_aud / num_batches,
            'fusion_loss': total_fus / num_batches,
            'fusion_acc': float(fusion_acc),
            'fusion_f1': float(fusion_f1),
            'audio_acc': float(audio_acc),
            'audio_f1': float(audio_f1)
        }

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, Any]):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': metrics,
            # Config elements needed to rebuild model
            'model_config': {
                'pretrained': False, # After training, checkpoint loading sets this False
            }
        }, str(path))
        
    def step(self, epoch: int) -> Dict[str, Any]:
        """
        Executes one full epoch: train -> validate -> save -> step scheduler.
        """
        train_metrics = self.train_epoch()
        val_metrics = self.validate()
        
        # Step LR scheduler based on TOTAL validation loss
        self.scheduler.step(val_metrics['loss'])
        
        # Save latest
        self.save_checkpoint(self.latest_checkpoint_path, epoch, val_metrics)
        
        # Save best
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(self.best_checkpoint_path, epoch, val_metrics)
            
        return {'train': train_metrics, 'val': val_metrics}
