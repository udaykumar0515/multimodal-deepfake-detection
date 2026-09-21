import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from preprocessing import VideoPreprocessor

DATASET_ROOT = project_root / "Dataset_FakeAVCeleb"
CSV_DIR = project_root / "data" / "dataset_split"
OUTPUT_DIR = project_root / "data" / "processed_frames"

def clear_line():
    sys.stdout.write('\033[K')

def print_status(split_name, current_idx, total_split, elapsed_time, 
                 total_success, total_failed, total_skipped, 
                 direct_det, forward_fills, center_fallbacks):
    
    total_processed = total_success + total_failed + total_skipped
    pct = (total_processed / 21544) * 100
    
    if total_success + total_failed > 0:
        speed = (total_success + total_failed) / elapsed_time
    else:
        speed = 0.0
        
    avg = (1.0 / speed) if speed > 0 else 0.0
    
    remaining = 21544 - total_processed
    eta_sec = remaining * avg
    
    eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m" if speed > 0 else "N/A"
    elapsed_str = f"{int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s"
    
    total_frames = (total_success * 16)
    if total_frames > 0:
        d_pct = (direct_det / total_frames) * 100
        f_pct = (forward_fills / total_frames) * 100
        c_pct = (center_fallbacks / total_frames) * 100
    else:
        d_pct = f_pct = c_pct = 0.0

    sys.stdout.write("\033[F" * 8)
    clear_line()
    print(f"[{split_name.upper()}] {current_idx}/{total_split} | Overall Progress: {pct:.2f}%")
    clear_line()
    print(f"Success: {total_success} | Skipped: {total_skipped} | Failed: {total_failed}")
    clear_line()
    print(f"Speed: {speed:.2f} videos/sec | Avg: {avg:.2f} sec/video")
    clear_line()
    print(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
    clear_line()
    print("Frames:")
    clear_line()
    print(f"Direct: {direct_det} ({d_pct:.1f}%) | Forward-fill: {forward_fills} ({f_pct:.1f}%) | Center fallback: {center_fallbacks} ({c_pct:.1f}%)")
    sys.stdout.flush()


def main(limit=None):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Starting Offline Video Preprocessing...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Initializing...")
    
    # Print 8 blank lines to make room for the updating block
    for _ in range(8):
        print()
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    
    if limit:
        train_df = train_df.head(limit)
        val_df = val_df.head(limit)
        test_df = test_df.head(limit)
        
    preprocessor = VideoPreprocessor(num_frames=16, margin=0.20, is_train=False)
    
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    direct_det = 0
    forward_fills = 0
    center_fallbacks = 0
    
    start_time = time.time()
    
    failure_log = []
    
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        for idx, row in enumerate(df.iterrows()):
            _, row_data = row
            sample_path_str = row_data['sample_path']
            video_full_path = DATASET_ROOT / sample_path_str
            
            # Format: .npy file
            path_without_ext = os.path.splitext(sample_path_str)[0]
            target_npy = OUTPUT_DIR / f"{path_without_ext}.npy"
            
            if os.path.exists(target_npy):
                total_skipped += 1
                if idx % 10 == 0 or idx == len(df) - 1:
                    elapsed = time.time() - start_time
                    print_status(split_name, idx + 1, len(df), elapsed, 
                                 total_success, total_failed, total_skipped, 
                                 direct_det, forward_fills, center_fallbacks)
                continue
                
            os.makedirs(os.path.dirname(target_npy), exist_ok=True)
            
            if not os.path.exists(video_full_path):
                total_failed += 1
                failure_log.append({"sample_path": sample_path_str, "error": "file_not_found"})
                continue
                
            raw_crops, failures = preprocessor.process(str(video_full_path), return_raw_crops=True)
            
            if raw_crops is None:
                total_failed += 1
                failure_log.append({"sample_path": sample_path_str, "error": failures[0]["reason"]})
            else:
                try:
                    # Convert list of 16 crops to a single uint8 array and save
                    stacked = np.stack(raw_crops).astype(np.uint8)
                    np.save(str(target_npy), stacked)
                    
                    total_success += 1
                    
                    # Tally failures (forward_filled vs center_fallback)
                    # Every frame is a direct detection unless it's in the failures list
                    video_ff = sum(1 for f in failures if f["reason"] == "forward_filled")
                    video_cf = sum(1 for f in failures if f["reason"] == "center_fallback" or f["reason"] == "no_face_detected")
                    video_direct = 16 - video_ff - video_cf
                    
                    direct_det += video_direct
                    forward_fills += video_ff
                    center_fallbacks += video_cf
                    
                except Exception as e:
                    total_failed += 1
                    failure_log.append({"sample_path": sample_path_str, "error": str(e)})
                    
            if idx % 5 == 0 or idx == len(df) - 1:
                elapsed = time.time() - start_time
                print_status(split_name, idx + 1, len(df), elapsed, 
                             total_success, total_failed, total_skipped, 
                             direct_det, forward_fills, center_fallbacks)

    print("\n\nOffline Preprocessing Complete.")
    
    with open(OUTPUT_DIR / "failures.json", "w") as f:
        json.dump(failure_log, f, indent=4)
        
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Success: {total_success}")
    print(f"Total Failed:  {total_failed}")
    if total_failed > 0:
        print(f"Failures saved to {OUTPUT_DIR / 'failures.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Only process a few videos")
    args = parser.parse_args()
    
    main(limit=3 if args.test_run else None)
