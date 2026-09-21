"""
train.py
========
Entry point for full multimodal deepfake detection training.

Usage:
    python scripts/train.py
    python scripts/train.py --resume checkpoints/latest.pt
    python scripts/train.py --epochs 20 --batch-size 4
"""

import os
import sys
import random
import argparse
import torch
import numpy as np
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from dataset.dataloader_factory import create_dataloaders
from training.trainer import Trainer

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# Conservative defaults for RTX 4050 6 GB VRAM.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "batch_size":       4,       # Safe for 6 GB VRAM with 16 × 224×224 video + audio
    "num_epochs":       10,
    "learning_rate":    1e-4,
    "weight_decay":     1e-4,
    "focal_gamma":      2.0,
    "num_workers":      4,
    "seed":             42,
    "checkpoint_dir":   str(project_root / "checkpoints"),
    "history_path":     str(project_root / "training_history.json"),
    "csv_dir":          str(project_root / "data" / "dataset_split"),
    "video_dir":        str(project_root / "data" / "processed_frames"),
    "audio_dir":        str(project_root / "data" / "processed_audio" / "spectrograms"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_banner(config: dict, device: torch.device) -> None:
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    print("==========================================")
    print(" Deepfake Detection — Multimodal Training")
    print("==========================================")
    print(f"\nDevice      : {device}")
    print(f"GPU         : {gpu_name}")
    print(f"\nDataset")
    print(f"  Train     : 15,083 videos")
    print(f"  Val       : 3,191  videos")
    print(f"  Test      : 3,270  videos")
    print(f"\nBatch size  : {config['batch_size']}")
    print(f"Epochs      : {config['num_epochs']}")
    print(f"LR          : {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Focal gamma : {config['focal_gamma']}")
    print(f"AMP         : {'enabled' if device.type == 'cuda' else 'disabled (CPU)'}")
    print(f"Checkpoints : {config['checkpoint_dir']}")
    print("==========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Deepfake Detection Training")
    parser.add_argument("--resume",     type=str,   default=None,   help="Path to checkpoint to resume from")
    parser.add_argument("--epochs",     type=int,   default=None,   help="Number of epochs")
    parser.add_argument("--batch-size", type=int,   default=None,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=None,   help="Learning rate")
    parser.add_argument("--gamma",      type=float, default=None,   help="Focal loss gamma")
    parser.add_argument("--seed",       type=int,   default=None,   help="Random seed")
    args = parser.parse_args()

    # ── Merge CLI overrides into config ────────────────────────────────────────
    config = dict(DEFAULT_CONFIG)
    if args.epochs     is not None: config["num_epochs"]    = args.epochs
    if args.batch_size is not None: config["batch_size"]    = args.batch_size
    if args.lr         is not None: config["learning_rate"] = args.lr
    if args.gamma      is not None: config["focal_gamma"]   = args.gamma
    if args.seed       is not None: config["seed"]          = args.seed

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_seed(config["seed"])

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_banner(config, device)

    # ── DataLoaders ────────────────────────────────────────────────────────────
    print("Loading DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(
        csv_dir=config["csv_dir"],
        video_dir=config["video_dir"],
        audio_dir=config["audio_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    print("Initialising model...")
    model = MultimodalDeepfakeModel(pretrained=True)

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # ── Resume ─────────────────────────────────────────────────────────────────
    if args.resume:
        trainer.resume(args.resume)

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()
