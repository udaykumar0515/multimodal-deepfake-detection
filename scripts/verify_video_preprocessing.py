import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path
import json

# Add project root to sys.path to import preprocessing
project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from preprocessing import VideoPreprocessor

DATASET_ROOT = project_root / "Dataset_FakeAVCeleb"
CSV_DIR = project_root / "data" / "dataset_split"
OUTPUT_DIR = project_root / "reports" / "preprocessing_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def verify_subset():
    print("Starting Video Preprocessing Verification...")
    
    # Load manifests
    try:
        train_df = pd.read_csv(CSV_DIR / "train.csv")
        val_df = pd.read_csv(CSV_DIR / "val.csv")
        test_df = pd.read_csv(CSV_DIR / "test.csv")
        print(f"Loaded manifests: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
    except Exception as e:
        print(f"Failed to load CSVs: {e}")
        return False
        
    # Pick a few samples
    samples = []
    if not train_df.empty: samples.append((train_df.iloc[0], "train"))
    if not val_df.empty: samples.append((val_df.iloc[0], "val"))
    if not test_df.empty: samples.append((test_df.iloc[0], "test"))
    
    # Optional: Pick one known to be long or short if we had duration, but we just take first ones.
    
    success_count = 0
    failure_count = 0
    
    preprocessor = VideoPreprocessor(num_frames=16, margin=0.20, is_train=False)
    
    for row, split in samples:
        sample_path = str(DATASET_ROOT / row['sample_path'])
        print(f"\nProcessing {split} sample: {row['sample_path']}")
        
        if not os.path.exists(sample_path):
            print(f"Error: Physical file not found at {sample_path}")
            failure_count += 1
            continue
            
        try:
            tensor, failures = preprocessor.process(sample_path)
            
            if tensor is None:
                print(f"Failed to process video: {failures}")
                failure_count += 1
                continue
                
            # Verification Checks
            assert tensor.shape == (16, 3, 224, 224), f"Unexpected shape {tensor.shape}"
            assert not torch.isnan(tensor).any(), "Tensor contains NaN"
            assert not torch.isinf(tensor).any(), "Tensor contains Inf"
            
            print(f"Verification PASS for {row['sample_path']}:")
            print(f"  - Output shape: {tensor.shape}")
            print(f"  - Min/Max values: {tensor.min().item():.3f} / {tensor.max().item():.3f}")
            print(f"  - Failures in video: {len(failures)}")
            if failures:
                print(f"    - First failure: {failures[0]}")
                
            # We would save debug images here, but since tensor is normalized, 
            # we just confirm the shapes are correct and the script runs without errors.
            success_count += 1
            
        except Exception as e:
            print(f"Exception during processing: {e}")
            failure_count += 1
            
    print(f"\nVerification Complete. Success: {success_count}, Failures: {failure_count}")
    
    # Write a small summary file
    with open(project_root / "reports" / "preprocessing_verification.txt", "w") as f:
        f.write(f"Verification Samples: {len(samples)}\n")
        f.write(f"Success Count: {success_count}\n")
        f.write(f"Failure Count: {failure_count}\n")
        
    return success_count == len(samples)

if __name__ == "__main__":
    verify_subset()
