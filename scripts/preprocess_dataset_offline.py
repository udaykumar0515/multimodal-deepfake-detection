import os
import sys
import pandas as pd
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from preprocessing import UniformTemporalSampler, RetinaFaceCropper

DATASET_ROOT = project_root / "Dataset_FakeAVCeleb"
CSV_DIR = project_root / "data" / "dataset_split"
OUTPUT_DIR = project_root / "data" / "processed_frames"

def process_video(sample_path_str, sampler, cropper):
    video_full_path = DATASET_ROOT / sample_path_str
    path_without_ext = os.path.splitext(sample_path_str)[0]
    target_dir = OUTPUT_DIR / path_without_ext
    
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) == 16:
        # Already processed
        return True

    if not os.path.exists(video_full_path):
        print(f"Error: Raw video not found at {video_full_path}")
        return False
        
    os.makedirs(target_dir, exist_ok=True)
    frames = sampler.sample(str(video_full_path))
    
    success = True
    for idx, frame in enumerate(frames):
        crop, info = cropper.crop_face(frame)
        
        if crop is None:
            # Fallback to center crop
            h, w, _ = frame.shape
            cw, ch = 224, 224
            x1, y1 = max(0, w//2 - cw//2), max(0, h//2 - ch//2)
            x2, y2 = min(w, x1 + cw), min(h, y1 + ch)
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size > 0:
                crop = cv2.resize(crop_img, (224, 224))
            else:
                crop = np.zeros((224, 224, 3), dtype=np.uint8)
                
        # Convert RGB to BGR for OpenCV saving
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        frame_filename = target_dir / f"frame_{idx:02d}.jpg"
        cv2.imwrite(str(frame_filename), crop_bgr)
        
    return success

def main(limit=None):
    print(f"Starting Offline Video Preprocessing...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load manifests
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    
    sampler = UniformTemporalSampler(num_frames=16)
    cropper = RetinaFaceCropper(margin=0.20, target_size=(224, 224))
    
    # If limit is provided for testing, just take the first few from each
    if limit:
        train_df = train_df.head(1)
        val_df = val_df.head(1)
        test_df = test_df.head(1)
        
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\nProcessing {split_name} Split ({len(df)} videos)...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            process_video(row['sample_path'], sampler, cropper)
            
    print("\nOffline Preprocessing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Only process 1 video per split (3 total)")
    args = parser.parse_args()
    
    main(limit=3 if args.test_run else None)
