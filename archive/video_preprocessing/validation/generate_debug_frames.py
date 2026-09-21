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
OUTPUT_DIR = project_root / "reports" / "preprocessing_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_samples():
    print("Generating Debug Frames...")
    
    # Load manifests
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    
    samples = []
    if not train_df.empty: samples.append((train_df.iloc[0], "train"))
    if not val_df.empty: samples.append((val_df.iloc[0], "val"))
    if not test_df.empty: samples.append((test_df.iloc[0], "test"))
    
    sampler = UniformTemporalSampler(num_frames=16)
    cropper = RetinaFaceCropper(margin=0.20)
    
    for row, split in samples:
        sample_path = str(DATASET_ROOT / row['sample_path'])
        print(f"Visualizing {split} sample: {row['sample_path']}")
        
        if not os.path.exists(sample_path):
            print(f"File not found: {sample_path}")
            continue
            
        frames = sampler.sample(sample_path)
        
        # Save a temporal summary grid of all 16 frames
        grid_h, grid_w = 4, 4
        # Calculate resize dims for the grid so it's not too huge
        th, tw = 120, 160
        grid_img = np.zeros((grid_h * th, grid_w * tw, 3), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            # Frame is RGB from sampler, convert to BGR for cv2 saving
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 1. Temporal grid
            row_idx = i // grid_w
            col_idx = i % grid_w
            small_frame = cv2.resize(frame_bgr, (tw, th))
            grid_img[row_idx*th:(row_idx+1)*th, col_idx*tw:(col_idx+1)*tw] = small_frame
            
            # Generate detailed debug for the first frame only to avoid clutter
            if i == 0:
                debug_frame = frame_bgr.copy()
                crop, info = cropper.crop_face(frame)
                
                if crop is not None:
                    # Draw original RetinaFace box (Blue)
                    ox1, oy1, ox2, oy2 = info["original_bbox"]
                    cv2.rectangle(debug_frame, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                    cv2.putText(debug_frame, "RetinaFace", (ox1, max(10, oy1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Draw 20% expanded box (Green)
                    nx1, ny1, nx2, ny2 = info["bbox"]
                    cv2.rectangle(debug_frame, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
                    cv2.putText(debug_frame, "Expanded (20%)", (nx1, min(debug_frame.shape[0]-10, ny2+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Save final crop (convert back to BGR for saving)
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(OUTPUT_DIR / f"{split}_frame0_final_crop.jpg"), crop_bgr)
                
                # Save debug annotated frame
                cv2.imwrite(str(OUTPUT_DIR / f"{split}_frame0_annotated.jpg"), debug_frame)
                
        # Save temporal grid
        cv2.imwrite(str(OUTPUT_DIR / f"{split}_temporal_sampling_grid.jpg"), grid_img)
        print(f"Saved debug images for {split} to {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_samples()
