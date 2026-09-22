import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd

from dataset.multimodal_dataset import MultimodalDeepfakeDataset

def create_dataloaders(csv_dir, video_dir, audio_dir, batch_size=4, num_workers=4):
    """
    Factory to create Train, Validation, and Test dataloaders.
    Implements a WeightedRandomSampler for the training loader to handle class imbalance.
    """
    train_csv = f"{csv_dir}/train.csv"
    val_csv = f"{csv_dir}/val.csv"
    test_csv = f"{csv_dir}/test.csv"
    
    train_dataset = MultimodalDeepfakeDataset(train_csv, video_dir, audio_dir, is_train=True)
    val_dataset = MultimodalDeepfakeDataset(val_csv, video_dir, audio_dir, is_train=False)
    test_dataset = MultimodalDeepfakeDataset(test_csv, video_dir, audio_dir, is_train=False)
    
    # Implement Imbalance-Aware Sampling for Train Split
    train_df = pd.read_csv(train_csv)
    labels = train_df['label'].str.strip().str.lower()
    
    real_count = (labels == 'real').sum()
    fake_count = (labels == 'fake').sum()
    
    weight_real = 1.0 / real_count if real_count > 0 else 0.0
    weight_fake = 1.0 / fake_count if fake_count > 0 else 0.0
    
    sample_weights = []
    for label in labels:
        if label == 'real':
            sample_weights.append(weight_real)
        else:
            sample_weights.append(weight_fake)
            
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights), # Maintains the original epoch length
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
