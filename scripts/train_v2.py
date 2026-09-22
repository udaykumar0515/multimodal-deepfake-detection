import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.multihead_model import MultiHeadDeepfakeModel
from dataset.dataloader_factory import create_dataloaders
from training.trainer import Trainer

def main():
    print("==========================================")
    print(" V2 Multi-Task Deepfake Training")
    print("==========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Configuration
    config = {
        "batch_size": 4,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "focal_gamma": 2.0,
        "gradient_clip": 1.0,
        "num_workers": 4,
        "csv_dir": str(project_root / "data" / "dataset_split"),
        "video_dir": str(project_root / "data" / "processed_frames"),
        "audio_dir": str(project_root / "data" / "processed_audio" / "spectrograms"),
        "checkpoint_dir": str(project_root / "checkpoints")
    }
    
    # 2. DataLoaders
    print("\n[1/3] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_dir=config["csv_dir"],
        video_dir=config["video_dir"],
        audio_dir=config["audio_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )
    print(f"  Train batches: {len(train_loader)} (WeightedRandomSampler active)")
    print(f"  Val batches:   {len(val_loader)} (Deterministic)")
    
    # 3. Model
    print("\n[2/3] Initializing V2 Multi-Head Model...")
    # pretrained=True loads ImageNet weights for both EfficientNet-B0 backbones
    model = MultiHeadDeepfakeModel(pretrained=True)
    
    # 4. Trainer
    print("\n[3/3] Initializing Multi-Task Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        focal_gamma=config["focal_gamma"],
        gradient_clip=config["gradient_clip"],
        device=device,
        checkpoint_dir=config["checkpoint_dir"]
    )
    
    print("\nSetup complete. Beginning 10-epoch training loop...")
    print(f"Best checkpoint will be saved to: {trainer.best_checkpoint_path}\n")
    
    # 5. Training Loop
    for epoch in range(1, config["num_epochs"] + 1):
        print(f"=== Epoch {epoch}/{config['num_epochs']} ===")
        metrics = trainer.step(epoch)
        
        train_m = metrics['train']
        val_m = metrics['val']
        
        print(f"  [Train] Total Loss: {train_m['loss']:.4f} | Img: {train_m['image_loss']:.4f} | Aud: {train_m['audio_loss']:.4f} | Fus: {train_m['fusion_loss']:.4f}")
        print(f"  [Val]   Total Loss: {val_m['loss']:.4f} | Img: {val_m['image_loss']:.4f} | Aud: {val_m['audio_loss']:.4f} | Fus: {val_m['fusion_loss']:.4f}")
        print(f"  [Val]   Fusion Acc: {val_m['fusion_acc']:.4f} | Fusion F1: {val_m['fusion_f1']:.4f} | Aud Acc: {val_m['audio_acc']:.4f}")
        
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
