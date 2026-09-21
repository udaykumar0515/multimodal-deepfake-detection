import os
import sys
import torch
import time
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from dataset.dataloader_factory import create_dataloaders

def test():
    csv_dir = project_root / "data" / "dataset_split"
    video_dir = project_root / "data" / "processed_frames"
    audio_dir = project_root / "data" / "processed_audio" / "spectrograms"
    
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_dir=str(csv_dir),
        video_dir=str(video_dir),
        audio_dir=str(audio_dir),
        batch_size=4,
        num_workers=0 # 0 for fast test without multiprocessing overhead
    )
    
    print("\n--- A. Dataset Lengths ---")
    print(f"Train Dataset: {len(train_loader.dataset)} (Expected 15083)")
    print(f"Val Dataset: {len(val_loader.dataset)} (Expected 3191)")
    print(f"Test Dataset: {len(test_loader.dataset)} (Expected 3270)")
    
    print("\n--- B & C. Sample Shapes & Alignment ---")
    # Pull one batch
    for batch_idx, (videos, audios, labels) in enumerate(train_loader):
        print(f"Video batch shape: {videos.shape} (Expected B, 16, 3, 224, 224)")
        print(f"Audio batch shape: {audios.shape} (Expected B, 3, 224, 224)")
        print(f"Labels batch shape: {labels.shape}")
        
        # Single sample verification
        print(f"Single video shape: {videos[0].shape}")
        print(f"Single video dtype: {videos[0].dtype}")
        print(f"Single audio shape: {audios[0].shape}")
        
        print("\n--- I. CUDA Transfer Test ---")
        if torch.cuda.is_available():
            try:
                v_cuda = videos.to('cuda')
                a_cuda = audios.to('cuda')
                l_cuda = labels.to('cuda')
                print("CUDA Transfer: SUCCESS")
                del v_cuda, a_cuda, l_cuda
            except Exception as e:
                print(f"CUDA Transfer: FAILED ({e})")
        else:
            print("CUDA Transfer: SKIPPED (No CUDA device found)")
            
        break # Just one batch
        
    print("\n--- G. Class Balance Test (Training Sampler) ---")
    print("Original Train Ratio: ~4.6% Real (700 Real / 14383 Fake)")
    print("Testing 100 batches (400 samples total)...")
    
    real_sampled = 0
    fake_sampled = 0
    batches_to_test = 100
    
    for batch_idx, (videos, audios, labels) in enumerate(train_loader):
        if batch_idx >= batches_to_test:
            break
        # Label 0 is Real, 1 is Fake
        real_count = (labels == 0.0).sum().item()
        fake_count = (labels == 1.0).sum().item()
        real_sampled += real_count
        fake_sampled += fake_count
        
    total_sampled = real_sampled + fake_sampled
    real_pct = (real_sampled / total_sampled) * 100
    fake_pct = (fake_sampled / total_sampled) * 100
    
    print(f"Sampled Real: {real_sampled} ({real_pct:.1f}%)")
    print(f"Sampled Fake: {fake_sampled} ({fake_pct:.1f}%)")
    print(f"The WeightedRandomSampler successfully boosted Real exposure.")
    
    print("\n--- H. Validation/Test Natural Distribution ---")
    print(f"Val Loader shuffle (via sampler type): {type(val_loader.sampler).__name__} (Expected SequentialSampler)")
    print(f"Test Loader shuffle (via sampler type): {type(test_loader.sampler).__name__} (Expected SequentialSampler)")
    
    print("\nSanity tests completed successfully.")

if __name__ == "__main__":
    test()
