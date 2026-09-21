import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from preprocessing.audio_preprocessing import AudioExtractor, SpectrogramGenerator

DATASET_ROOT = project_root / "Dataset_FakeAVCeleb"
CSV_DIR = project_root / "data" / "dataset_split"
AUDIO_OUT_DIR = project_root / "data" / "processed_audio" / "raw_wav"
SPEC_OUT_DIR = project_root / "data" / "processed_audio" / "spectrograms"

def clear_line():
    sys.stdout.write('\033[K')

def print_status(split_name, current_idx, total_split, elapsed_time, 
                 total_success, total_failed, total_skipped):
    
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

    sys.stdout.write("\033[F" * 5)
    clear_line()
    print(f"[{split_name.upper()}] {current_idx}/{total_split} | Overall Progress: {pct:.2f}%")
    clear_line()
    print(f"Success: {total_success} | Skipped: {total_skipped} | Failed: {total_failed}")
    clear_line()
    print(f"Speed: {speed:.2f} videos/sec | Avg: {avg:.2f} sec/video")
    clear_line()
    print(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
    clear_line()
    print("-" * 50)
    sys.stdout.flush()

def main(limit=None):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Starting Offline Audio Preprocessing...")
    print(f"WAV Output directory: {AUDIO_OUT_DIR}")
    print(f"Spec Output directory: {SPEC_OUT_DIR}")
    print("Initializing...")
    
    for _ in range(5):
        print()
        
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(SPEC_OUT_DIR, exist_ok=True)
    
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    
    if limit:
        train_df = train_df.head(limit)
        val_df = val_df.head(limit)
        test_df = test_df.head(limit)
        
    extractor = AudioExtractor()
    generator = SpectrogramGenerator()
    
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    start_time = time.time()
    failure_log = []
    
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        for idx, row in enumerate(df.iterrows()):
            _, row_data = row
            sample_path_str = row_data['sample_path']
            video_full_path = DATASET_ROOT / sample_path_str
            
            path_without_ext = os.path.splitext(sample_path_str)[0]
            target_wav = AUDIO_OUT_DIR / f"{path_without_ext}.wav"
            target_npy = SPEC_OUT_DIR / f"{path_without_ext}.npy"
            
            if os.path.exists(target_wav) and os.path.exists(target_npy):
                total_skipped += 1
                if idx % 10 == 0 or idx == len(df) - 1:
                    elapsed = time.time() - start_time
                    print_status(split_name, idx + 1, len(df), elapsed, 
                                 total_success, total_failed, total_skipped)
                continue
                
            os.makedirs(os.path.dirname(target_wav), exist_ok=True)
            os.makedirs(os.path.dirname(target_npy), exist_ok=True)
            
            if not os.path.exists(video_full_path):
                total_failed += 1
                failure_log.append({"sample_path": sample_path_str, "error": "video_file_not_found"})
                continue
                
            try:
                # 1. Extract WAV
                if not os.path.exists(target_wav):
                    extractor.extract(str(video_full_path), str(target_wav))
                    
                # 2. Generate Spectrogram
                spec_tensor = generator.generate(str(target_wav))
                
                # 3. Save as NPY (float32, [3, 224, 224])
                spec_np = spec_tensor.numpy().astype(np.float32)
                np.save(str(target_npy), spec_np)
                
                total_success += 1
                
            except Exception as e:
                total_failed += 1
                failure_log.append({"sample_path": sample_path_str, "error": str(e)})
                
            if idx % 5 == 0 or idx == len(df) - 1:
                elapsed = time.time() - start_time
                print_status(split_name, idx + 1, len(df), elapsed, 
                             total_success, total_failed, total_skipped)

    print("\n\nOffline Audio Preprocessing Complete.")
    
    with open(project_root / "data" / "processed_audio" / "failures.json", "w") as f:
        json.dump(failure_log, f, indent=4)
        
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Success: {total_success}")
    print(f"Total Failed:  {total_failed}")
    if total_failed > 0:
        print("Failures saved to failures.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Only process a few videos")
    args = parser.parse_args()
    
    main(limit=3 if args.test_run else None)
