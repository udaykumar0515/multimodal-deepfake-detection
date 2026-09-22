import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultimodalDeepfakeDataset(Dataset):
    """
    PyTorch Dataset for V2 Multi-Head Multimodal Deepfake Detection.
    Lazily loads processed .npy video frames and audio spectrograms.
    Applies temporally consistent spatial augmentations to the 16 video frames.
    
    Returns:
        video_tensor: (16, 3, 224, 224)
        audio_tensor: (3, 224, 224)
        video_label: (1,) - Target for the Image Head
        audio_label: (1,) - Target for the Audio Head
        overall_label: (1,) - Target for the Fusion Head
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

    def _parse_label(self, label_str, row_idx, column_name):
        label_str = str(label_str).strip().lower()
        if label_str == 'real':
            return 0.0
        elif label_str == 'fake':
            return 1.0
        else:
            raise ValueError(f"Unknown label '{label_str}' in column '{column_name}' for row {row_idx}")

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
        
        # Parse all three labels
        v_lbl = self._parse_label(row['video_label'], idx, 'video_label')
        a_lbl = self._parse_label(row['audio_label'], idx, 'audio_label')
        o_lbl = self._parse_label(row['label'], idx, 'label')
            
        video_label_tensor = torch.tensor([v_lbl], dtype=torch.float32)
        audio_label_tensor = torch.tensor([a_lbl], dtype=torch.float32)
        overall_label_tensor = torch.tensor([o_lbl], dtype=torch.float32)
        
        return video_tensor, audio_tensor, video_label_tensor, audio_label_tensor, overall_label_tensor
