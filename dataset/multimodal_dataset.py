import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultimodalDeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Deepfake Detection.
    Lazily loads processed .npy video frames and audio spectrograms.
    Applies temporally consistent spatial augmentations to the 16 video frames.
    """
    def __init__(self, csv_path, video_dir, audio_dir, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.is_train = is_train
        
        # Albumentations setup for consistent 16-frame augmentation
        # We map 'image' -> frame 0, and 'image1' to 'image15' -> frames 1 to 15
        additional_targets = {f'image{i}': 'image' for i in range(1, 16)}
        
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                # A.CoarseDropout doesn't natively support additional_targets perfectly for all versions,
                # but A.CoarseDropout is deprecated in 2.x in favor of A.PixelDropout/Erasing.
                # However, RandomCrop/transforms are safer. We will use Erasing if available, 
                # but for simplicity and robust temporal consistency across albumentations versions,
                # we'll stick to the core geometric and color ones here, plus standard normalization.
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets=additional_targets)
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets=additional_targets)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_path = row['sample_path']
        base_name = os.path.splitext(sample_path)[0]
        
        video_path = os.path.join(self.video_dir, f"{base_name}.npy")
        audio_path = os.path.join(self.audio_dir, f"{base_name}.npy")
        
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Missing processed file for {sample_path}\n"
                f"Expected Video: {video_path}\n"
                f"Expected Audio: {audio_path}"
            )
            
        # Load video: (16, 224, 224, 3) uint8
        video_np = np.load(video_path)
        
        # Prepare kwargs for albumentations
        kwargs = {'image': video_np[0]}
        for i in range(1, 16):
            kwargs[f'image{i}'] = video_np[i]
            
        transformed = self.transform(**kwargs)
        
        # Reconstruct into (16, 3, 224, 224) float32
        frames = [transformed['image']]
        for i in range(1, 16):
            frames.append(transformed[f'image{i}'])
        
        video_tensor = torch.stack(frames) # (16, 3, 224, 224)
        
        # Load audio: (3, 224, 224) float32
        audio_np = np.load(audio_path)
        audio_tensor = torch.from_numpy(audio_np).float()
        
        # Label mapping: Real -> 0.0, Fake -> 1.0
        label_str = str(row['label']).strip().lower()
        if label_str == 'real':
            label_val = 0.0
        elif label_str == 'fake':
            label_val = 1.0
        else:
            raise ValueError(f"Unknown label {label_str} in row {idx}")
            
        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        
        return video_tensor, audio_tensor, label_tensor
