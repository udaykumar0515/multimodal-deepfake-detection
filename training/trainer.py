"""
Trainer
=======
Encapsulates the full training + validation loop for the multimodal
deepfake detection model.

Responsibilities:
    - Training loop with AMP/GradScaler on CUDA
    - Validation loop (eval mode, no grad)
    - Focal Loss integration
    - Epoch-level progress display via tqdm
    - Checkpoint saving (latest + best)
    - Resume from checkpoint
    - Training history JSON logging
    - Learning-rate scheduler step
"""

import os
import json
import time
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import BinaryFocalLoss


class Trainer:
    """
    Args:
        model:           MultimodalDeepfakeModel instance
        train_loader:    Training DataLoader (with WeightedRandomSampler)
        val_loader:      Validation DataLoader
        config (dict):   Training hyper-parameters and paths
        device:          torch.device
    """
    def __init__(self, model, train_loader: DataLoader,
                 val_loader: DataLoader, config: dict,
                 device: torch.device) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = BinaryFocalLoss(gamma=config["focal_gamma"])

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # ── Scheduler (CosineAnnealingLR) ─────────────────────────────────────
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["num_epochs"],
            eta_min=config["learning_rate"] * 0.01,
        )

        # ── AMP ───────────────────────────────────────────────────────────────
        self.use_amp = (device.type == "cuda")
        self.scaler  = torch.amp.GradScaler("cuda") if self.use_amp else None

        # ── State ─────────────────────────────────────────────────────────────
        self.start_epoch   = 0
        self.best_val_loss = float("inf")
        self.history       = []

        # ── Paths ─────────────────────────────────────────────────────────────
        self.ckpt_dir      = config["checkpoint_dir"]
        self.history_path  = config.get("history_path", "training_history.json")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the complete training loop for config['num_epochs'] epochs."""
        num_epochs = self.config["num_epochs"]

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            train_loss = self._train_epoch(epoch, num_epochs)
            val_loss   = self._validate_epoch(epoch, num_epochs)

            self.scheduler.step()

            lr_now     = self.scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start

            # ── Epoch summary ──────────────────────────────────────────────
            print(
                f"\nEpoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {lr_now:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # ── History ────────────────────────────────────────────────────
            self.history.append({
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         lr_now,
                "time_sec":   round(epoch_time, 2),
            })
            self._save_history()

            # ── Checkpointing ──────────────────────────────────────────────
            self._save_checkpoint("latest.pt", epoch + 1, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt", epoch + 1, val_loss)
                print(f"  [BEST] New best val loss: {val_loss:.4f}  ->  best_model.pt saved")

        print("\n==========================================")
        print("Training complete.")
        print(f"Best validation loss : {self.best_val_loss:.4f}")
        print(f"Best checkpoint      : {os.path.join(self.ckpt_dir, 'best_model.pt')}")
        print("==========================================")

    def resume(self, checkpoint_path: str) -> None:
        """Load state from checkpoint_path and resume from the correct epoch."""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch   = ckpt["epoch"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.history       = ckpt.get("history", [])

        if self.scaler is not None and "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])

        print(f"  Resumed at epoch {self.start_epoch + 1}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{total_epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        for video, audio, labels in pbar:
            video  = video.to(self.device, non_blocking=True)
            audio  = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(video, audio)
                    loss   = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(video, audio)
                loss   = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            loss_val = loss.detach().item()
            total_loss += loss_val
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        return total_loss / max(n_batches, 1)

    def _validate_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{total_epochs} [Val  ]",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for video, audio, labels in pbar:
                video  = video.to(self.device, non_blocking=True)
                audio  = audio.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(video, audio)
                        loss   = self.criterion(logits, labels)
                else:
                    logits = self.model(video, audio)
                    loss   = self.criterion(logits, labels)

                total_loss += loss.item()
                n_batches  += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss":   self.best_val_loss,
            "history":         self.history,
            "config":          self.config,
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()

        path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, path)

    def _save_history(self) -> None:
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)
