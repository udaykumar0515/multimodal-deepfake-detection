import os
import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent.parent.parent
sys.path.append(str(project_root))

from preprocessing import UniformTemporalSampler, RetinaFaceCropper

DATASET_ROOT = project_root / "Dataset_FakeAVCeleb"
CSV_DIR = project_root / "data" / "dataset_split"
OUTPUT_DIR = project_root / "data" / "processed_faces_sample"

def simulate_offline_processing():
    print("Simulating Offline Preprocessing for 3 Sample Videos...")
    
    # Load manifests
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    
    samples = []
    if not train_df.empty: samples.append((train_df.iloc[0], "train"))
    if not val_df.empty: samples.append((val_df.iloc[0], "val"))
    if not test_df.empty: samples.append((test_df.iloc[0], "test"))
    
    sampler = UniformTemporalSampler(num_frames=16)
    cropper = RetinaFaceCropper(margin=0.20, target_size=(224, 224))
    
    for row, split in samples:
        sample_path_str = row['sample_path']
        video_full_path = DATASET_ROOT / sample_path_str
        
        # Determine the target directory mirroring the raw dataset structure
        # e.g., FakeVideo-FakeAudio/African/men/id00166/00010_id00366_wavtolip.mp4
        # becomes .../processed_faces_sample/FakeVideo-FakeAudio/African/men/id00166/00010_id00366_wavtolip/
        path_without_ext = os.path.splitext(sample_path_str)[0]
        target_dir = OUTPUT_DIR / path_without_ext
        
        os.makedirs(target_dir, exist_ok=True)
        print(f"\nProcessing [{split}] video: {sample_path_str}")
        print(f"Target Directory: {target_dir}")
        
        if not os.path.exists(video_full_path):
            print(f"Error: Raw video not found at {video_full_path}")
            continue
            
        frames = sampler.sample(str(video_full_path))
        
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
            
        print(f"Successfully saved 16 face crops to: {target_dir}")

if __name__ == "__main__":
    simulate_offline_processing()
