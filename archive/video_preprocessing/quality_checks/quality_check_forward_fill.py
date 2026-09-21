import os
import sys
import time
import pandas as pd
import numpy as np
import cv2
import json
import math
from pathlib import Path
from tqdm import tqdm

project_root = Path(os.path.abspath(__file__)).parent.parent.parent.parent
sys.path.append(str(project_root))

from preprocessing import UniformTemporalSampler, RetinaFaceCropper
from preprocessing.video_preprocessing import HAVE_INSIGHTFACE

DATASET_ROOT = project_root / "Dataset_FakeAVCeleb"
CSV_DIR = project_root / "data" / "dataset_split"
OUTPUT_DIR = project_root / "reports" / "forward_fill_quality"
VISUAL_DIR = OUTPUT_DIR / "visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

def run_diagnostic():
    print("Starting Forward-Fill Quality Diagnostic...")
    
    np.random.seed(42)
    
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    
    def sample_mix(df, n):
        if 'label' in df.columns:
            real_df = df[df['label'] == 'Real']
            fake_df = df[df['label'] == 'Fake']
            n_real = min(len(real_df), n // 2)
            n_fake = n - n_real
            s_real = real_df.sample(n=n_real, random_state=42) if not real_df.empty else pd.DataFrame()
            s_fake = fake_df.sample(n=n_fake, random_state=42) if not fake_df.empty else pd.DataFrame()
            return pd.concat([s_real, s_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            return df.sample(n=n, random_state=42)

    sample_train = sample_mix(train_df, 25)
    sample_val = sample_mix(val_df, 15)
    sample_test = sample_mix(test_df, 10)
    
    all_samples = pd.concat([sample_train, sample_val, sample_test]).reset_index(drop=True)
    
    sampler = UniformTemporalSampler(num_frames=16)
    cropper = RetinaFaceCropper(margin=0.20, target_size=(224, 224))
    
    stats = []
    all_ff_streaks = []
    all_drifts = []
    
    # Store annotated frames to dump later for selected representatives
    video_frames_cache = {} 
    
    for idx, row in tqdm(all_samples.iterrows(), total=len(all_samples)):
        sample_path_str = row['sample_path']
        video_full_path = DATASET_ROOT / sample_path_str
        
        video_stat = {
            "index": idx,
            "path": sample_path_str,
            "detections": 0,
            "forward_fills": 0,
            "center_fallbacks": 0,
            "max_ff_streak": 0,
            "mean_ff_streak": 0,
            "drift_avg": 0,
            "error": None
        }
        
        if not os.path.exists(video_full_path):
            video_stat["error"] = "file_not_found"
            stats.append(video_stat)
            continue
            
        try:
            frames = sampler.sample(str(video_full_path))
            
            last_valid_bbox = None
            current_ff_streak = 0
            streaks = []
            drifts = []
            
            annotated_frames = []
            
            for f_idx, frame in enumerate(frames):
                crop, info = cropper.crop_face(frame)
                
                is_fallback = False
                is_forward_fill = False
                final_crop = None
                
                if crop is None:
                    if last_valid_bbox is not None:
                        final_crop, ff_info = cropper.apply_bbox_and_crop(frame, last_valid_bbox)
                        if final_crop is not None:
                            is_forward_fill = True
                            video_stat["forward_fills"] += 1
                            current_ff_streak += 1
                        else:
                            is_fallback = True
                            video_stat["center_fallbacks"] += 1
                            if current_ff_streak > 0:
                                streaks.append(current_ff_streak)
                            current_ff_streak = 0
                    else:
                        is_fallback = True
                        video_stat["center_fallbacks"] += 1
                        
                    if is_fallback:
                        h, w, _ = frame.shape
                        cw, ch = 224, 224
                        x1, y1 = max(0, w//2 - cw//2), max(0, h//2 - ch//2)
                        x2, y2 = min(w, x1 + cw), min(h, y1 + ch)
                        crop_img = frame[y1:y2, x1:x2]
                        if crop_img.size > 0:
                            final_crop = cv2.resize(crop_img, (224, 224))
                        else:
                            final_crop = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    video_stat["detections"] += 1
                    
                    if current_ff_streak > 0:
                        streaks.append(current_ff_streak)
                        # Measure drift from the last FF bbox center to the new detected bbox center
                        prev_x1, prev_y1, prev_x2, prev_y2 = last_valid_bbox
                        prev_cx, prev_cy = (prev_x1+prev_x2)/2, (prev_y1+prev_y2)/2
                        
                        curr_x1, curr_y1, curr_x2, curr_y2 = info["original_bbox"]
                        curr_cx, curr_cy = (curr_x1+curr_x2)/2, (curr_y1+curr_y2)/2
                        
                        dist = math.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                        diag = math.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                        drift_pct = (dist / diag) * 100
                        drifts.append(drift_pct)
                        
                        current_ff_streak = 0
                        
                    last_valid_bbox = info["original_bbox"]
                    final_crop = crop
                
                # Render Visual
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                crop_bgr = cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR)
                annotated = frame_bgr.copy()
                
                if not is_fallback and not is_forward_fill:
                    ox1, oy1, ox2, oy2 = info["original_bbox"]
                    cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                    nx1, ny1, nx2, ny2 = info["bbox"]
                    cv2.rectangle(annotated, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
                    status_text = "DIRECT"
                    color = (0, 255, 0)
                elif is_forward_fill:
                    ox1, oy1, ox2, oy2 = last_valid_bbox
                    cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                    nx1, ny1, nx2, ny2 = ff_info["bbox"]
                    cv2.rectangle(annotated, (nx1, ny1), (nx2, ny2), (0, 255, 255), 2)
                    status_text = "FORWARD_FILL"
                    color = (0, 255, 255)
                else:
                    status_text = "CENTER_FALLBACK"
                    color = (0, 0, 255)
                    
                cv2.putText(annotated, f"Frame {f_idx}: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                h, w, _ = annotated.shape
                scale = 224 / h
                new_w = int(w * scale)
                annotated_resized = cv2.resize(annotated, (new_w, 224))
                combined = np.hstack((annotated_resized, crop_bgr))
                
                annotated_frames.append(combined)
            
            if current_ff_streak > 0:
                streaks.append(current_ff_streak)
                
            video_stat["max_ff_streak"] = max(streaks) if streaks else 0
            video_stat["mean_ff_streak"] = sum(streaks)/len(streaks) if streaks else 0
            video_stat["drift_avg"] = sum(drifts)/len(drifts) if drifts else 0
            
            video_frames_cache[idx] = annotated_frames
            
            if streaks:
                all_ff_streaks.extend(streaks)
            if drifts:
                all_drifts.extend(drifts)
                
        except Exception as e:
            video_stat["error"] = str(e)
            
        stats.append(video_stat)
        
    df_stats = pd.DataFrame(stats)
    
    # Identify representatives
    reps = {}
    
    # 1. 0 direct detections
    rep_0_det = df_stats[(df_stats['detections'] == 0) & (df_stats['error'].isnull())]
    if not rep_0_det.empty:
        reps["0_detections"] = int(rep_0_det.iloc[0]['index'])
        
    # 2. Long sequence of FF (max streak)
    rep_long_ff = df_stats[df_stats['error'].isnull()].sort_values('max_ff_streak', ascending=False)
    if not rep_long_ff.empty:
        reps["longest_ff_streak"] = int(rep_long_ff.iloc[0]['index'])
        
    # 3. Many direct + some FF
    rep_mixed = df_stats[(df_stats['detections'] > 5) & (df_stats['forward_fills'] > 2) & (df_stats['error'].isnull())].sort_values('detections', ascending=False)
    if not rep_mixed.empty:
        reps["mixed_direct_and_ff"] = int(rep_mixed.iloc[0]['index'])
        
    # 4. Mostly FF
    rep_mostly_ff = df_stats[(df_stats['forward_fills'] > 10) & (df_stats['error'].isnull())].sort_values('forward_fills', ascending=False)
    if not rep_mostly_ff.empty:
        reps["mostly_ff"] = int(rep_mostly_ff.iloc[0]['index'])
        
    # 5. Substantial center fallback
    rep_center = df_stats[(df_stats['center_fallbacks'] > 8) & (df_stats['detections'] > 0) & (df_stats['error'].isnull())].sort_values('center_fallbacks', ascending=False)
    if not rep_center.empty:
        reps["high_center_fallback"] = int(rep_center.iloc[0]['index'])
        
    # Dump visuals for reps
    for name, v_idx in reps.items():
        frames = video_frames_cache.get(v_idx, [])
        for f_idx, f_img in enumerate(frames):
            safe_name = f"{name}_vid{v_idx}_f{f_idx}.jpg"
            cv2.imwrite(str(VISUAL_DIR / safe_name), f_img)
            
    summary = {
        "overall_ff_streaks": {
            "max": int(max(all_ff_streaks)) if all_ff_streaks else 0,
            "mean": float(np.mean(all_ff_streaks)) if all_ff_streaks else 0,
            "count": len(all_ff_streaks)
        },
        "overall_drift_pct": {
            "max": float(max(all_drifts)) if all_drifts else 0,
            "mean": float(np.mean(all_drifts)) if all_drifts else 0
        },
        "representatives_saved": reps
    }
    
    with open(OUTPUT_DIR / "quality_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("\nQuality Diagnostic Complete.")
    print(json.dumps(summary, indent=4))
    
if __name__ == "__main__":
    run_diagnostic()
