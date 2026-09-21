import os
import sys
import time
import pandas as pd
import numpy as np
import cv2
import torch
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
import json

project_root = Path(os.path.abspath(__file__)).parent.parent.parent.parent
sys.path.append(str(project_root))

from preprocessing import UniformTemporalSampler, RetinaFaceCropper
from preprocessing.video_preprocessing import HAVE_INSIGHTFACE

DATASET_ROOT = project_root / "Dataset_FakeAVCeleb"
CSV_DIR = project_root / "data" / "dataset_split"
OUTPUT_DIR = project_root / "reports" / "video_preprocessing_validation"
VISUAL_DIR = OUTPUT_DIR / "visual_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

def run_validation():
    print("Starting Video Preprocessing Quality Validation...")
    
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if HAVE_INSIGHTFACE:
        print(f"ONNX Runtime Providers: {ort.get_available_providers()}")
    else:
        print("InsightFace not installed!")
        
    # Set fixed seed
    np.random.seed(42)
    
    # Load manifests
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    
    # Sample videos: 25 train, 15 val, 10 test (Total 50)
    # Mix of real and fake if possible
    
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
    
    sample_train['split'] = 'train'
    sample_val['split'] = 'val'
    sample_test['split'] = 'test'
    
    all_samples = pd.concat([sample_train, sample_val, sample_test]).reset_index(drop=True)
    
    print(f"Selected {len(all_samples)} videos for validation.")
    
    sampler = UniformTemporalSampler(num_frames=16)
    
    # Let's verify what provider insightface is actually using. 
    # By default RetinaFaceCropper uses CPUExecutionProvider in our code.
    cropper = RetinaFaceCropper(margin=0.20, target_size=(224, 224))
    
    stats = []
    
    # We will save visuals for the first 5 videos
    num_visuals_to_save = 5
    visuals_saved = 0
    
    total_start_time = time.time()
    total_frames_processed = 0
    
    for idx, row in tqdm(all_samples.iterrows(), total=len(all_samples)):
        sample_path_str = row['sample_path']
        video_full_path = DATASET_ROOT / sample_path_str
        
        video_stat = {
            "index": idx,
            "split": row['split'],
            "path": sample_path_str,
            "label": row.get('label', 'Unknown'),
            "detections": 0,
            "forward_fills": 0,
            "center_fallbacks": 0,
            "processing_time": 0.0,
            "error": None,
            "frames_processed": 0
        }
        
        if not os.path.exists(video_full_path):
            video_stat["error"] = "file_not_found"
            stats.append(video_stat)
            continue
            
        start_time = time.time()
        
        try:
            frames = sampler.sample(str(video_full_path))
            video_stat["frames_processed"] = len(frames)
            total_frames_processed += len(frames)
            
            save_visuals = visuals_saved < num_visuals_to_save
            if save_visuals:
                # Create a grid for this video: 16 rows (frames), 3 cols (original, detection bbox, final crop)
                # To keep it small, maybe just save the first 4 frames
                pass
                
            last_valid_bbox = None
            
            for f_idx, frame in enumerate(frames):
                crop, info = cropper.crop_face(frame)
                
                is_fallback = False
                is_forward_fill = False
                final_crop = None
                
                if crop is None:
                    if last_valid_bbox is not None:
                        # Forward fill
                        final_crop, ff_info = cropper.apply_bbox_and_crop(frame, last_valid_bbox)
                        if final_crop is not None:
                            is_forward_fill = True
                            video_stat["forward_fills"] += 1
                        else:
                            is_fallback = True
                            video_stat["center_fallbacks"] += 1
                    else:
                        is_fallback = True
                        video_stat["center_fallbacks"] += 1
                        
                    if is_fallback:
                        # Center fallback
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
                    last_valid_bbox = info["original_bbox"]
                    final_crop = crop
                    
                if save_visuals and f_idx < 4: # Save visuals for first 4 frames
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    crop_bgr = cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR)
                    
                    # Annotated frame
                    annotated = frame_bgr.copy()
                    if not is_fallback and not is_forward_fill:
                        ox1, oy1, ox2, oy2 = info["original_bbox"]
                        cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                        nx1, ny1, nx2, ny2 = info["bbox"]
                        cv2.rectangle(annotated, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
                        status_text = "Detected"
                        color = (0, 255, 0)
                    elif is_forward_fill:
                        ox1, oy1, ox2, oy2 = last_valid_bbox
                        cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                        nx1, ny1, nx2, ny2 = ff_info["bbox"]
                        cv2.rectangle(annotated, (nx1, ny1), (nx2, ny2), (0, 255, 255), 2) # Yellow
                        status_text = "Forward_Filled"
                        color = (0, 255, 255)
                    else:
                        status_text = "Center_Fallback"
                        color = (0, 0, 255)
                        
                    cv2.putText(annotated, f"Frame {f_idx}: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Save side-by-side
                    # Resize annotated to have same height as crop (224)
                    h, w, _ = annotated.shape
                    scale = 224 / h
                    new_w = int(w * scale)
                    annotated_resized = cv2.resize(annotated, (new_w, 224))
                    
                    combined = np.hstack((annotated_resized, crop_bgr))
                    
                    safe_name = sample_path_str.replace('/', '_').replace('\\', '_')
                    cv2.imwrite(str(VISUAL_DIR / f"vid{idx}_f{f_idx}_{status_text}.jpg"), combined)
                    
            if save_visuals:
                visuals_saved += 1
                
        except Exception as e:
            video_stat["error"] = str(e)
            
        end_time = time.time()
        video_stat["processing_time"] = end_time - start_time
        
        stats.append(video_stat)
        
    total_end_time = time.time()
    
    # Calculate aggregate stats
    df_stats = pd.DataFrame(stats)
    
    # Save raw stats
    df_stats.to_csv(OUTPUT_DIR / "validation_stats.csv", index=False)
    
    # Summary
    successful_videos = df_stats[df_stats['error'].isnull()]
    
    total_videos = len(successful_videos)
    total_detections = successful_videos['detections'].sum()
    total_forward_fills = successful_videos['forward_fills'].sum()
    total_center_fallbacks = successful_videos['center_fallbacks'].sum()
    total_processed_frames = successful_videos['frames_processed'].sum()
    
    avg_processing_time = successful_videos['processing_time'].mean()
    total_processing_time = total_end_time - total_start_time
    
    detection_rate = total_detections / total_processed_frames if total_processed_frames > 0 else 0
    forward_fill_rate = total_forward_fills / total_processed_frames if total_processed_frames > 0 else 0
    center_fallback_rate = total_center_fallbacks / total_processed_frames if total_processed_frames > 0 else 0
    
    zero_detections = len(successful_videos[successful_videos['detections'] == 0])
    under_8_detections = len(successful_videos[successful_videos['detections'] < 8])
    over_50pct_center_fallback = len(successful_videos[successful_videos['center_fallbacks'] > 8])
    zero_center_fallbacks = len(successful_videos[successful_videos['center_fallbacks'] == 0])
    
    summary = {
        "hardware": {
            "cuda_available": cuda_available,
            "onnx_providers": ort.get_available_providers() if HAVE_INSIGHTFACE else []
        },
        "totals": {
            "videos_tested": total_videos,
            "frames_processed": int(total_processed_frames),
            "total_detections": int(total_detections),
            "total_forward_fills": int(total_forward_fills),
            "total_center_fallbacks": int(total_center_fallbacks),
        },
        "rates": {
            "detection_success_rate": float(detection_rate),
            "forward_fill_rate": float(forward_fill_rate),
            "center_fallback_rate": float(center_fallback_rate)
        },
        "problematic_videos": {
            "zero_successful_detections": zero_detections,
            "fewer_than_8_detections": under_8_detections,
            "more_than_50pct_center_fallback": over_50pct_center_fallback,
            "zero_center_fallbacks": zero_center_fallbacks
        },
        "per_video_stats": {
            "min_detections": int(successful_videos['detections'].min()),
            "max_detections": int(successful_videos['detections'].max()),
            "mean_detections": float(successful_videos['detections'].mean()),
            "min_forward_fills": int(successful_videos['forward_fills'].min()),
            "max_forward_fills": int(successful_videos['forward_fills'].max()),
            "mean_forward_fills": float(successful_videos['forward_fills'].mean()),
            "min_center_fallbacks": int(successful_videos['center_fallbacks'].min()),
            "max_center_fallbacks": int(successful_videos['center_fallbacks'].max()),
            "mean_center_fallbacks": float(successful_videos['center_fallbacks'].mean())
        },
        "timing": {
            "avg_processing_time_per_video": float(avg_processing_time),
            "avg_processing_time_per_frame": float(avg_processing_time / 16) if total_videos > 0 else 0,
            "total_validation_time": float(total_processing_time)
        }
    }
    
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("\nValidation Complete. Generating report data...")
    print(json.dumps(summary, indent=4))
    
if __name__ == "__main__":
    run_validation()
