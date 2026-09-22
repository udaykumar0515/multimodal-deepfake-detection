import os
import shutil
import pandas as pd
import cv2
import subprocess
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
test_csv_path = project_root / "data" / "dataset_split" / "test.csv"
videos_base_dir = project_root / "archive" / "Dataset_FakeAVCeleb"
testing_data_dir = project_root / "testing_data"

import sys
sys.path.append(str(project_root))
from preprocessing.audio_preprocessing import AudioExtractor

audio_extractor = AudioExtractor()

def extract_audio(video_path, audio_path):
    audio_extractor.extract(str(video_path), str(audio_path))

def extract_frame(video_path, image_path):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        # Get frame in the middle
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(image_path), frame)
    cap.release()

def main():
    df = pd.read_csv(test_csv_path)
    manifest_rows = []
    
    categories = [
        "RealVideo-RealAudio",
        "FakeVideo-RealAudio",
        "RealVideo-FakeAudio",
        "FakeVideo-FakeAudio"
    ]
    
    for cat in categories:
        cat_df = df[df['type'] == cat].head(4) # 4 samples per category
        for idx, row in cat_df.iterrows():
            source_path = videos_base_dir / row['sample_path']
            
            # 1. Video
            cat_map = {
                "RealVideo-RealAudio": "real_video_real_audio",
                "FakeVideo-RealAudio": "fake_video_real_audio",
                "RealVideo-FakeAudio": "real_video_fake_audio",
                "FakeVideo-FakeAudio": "fake_video_fake_audio"
            }
            vid_cat_name = cat_map[cat]
            vid_filename = f"{cat}_{idx}.mp4"
            dest_vid_path = testing_data_dir / "videos" / vid_cat_name / vid_filename
            if source_path.exists():
                shutil.copy2(source_path, dest_vid_path)
                manifest_rows.append({
                    "modality": "video",
                    "category": cat,
                    "original_path": str(row['sample_path']),
                    "testing_path": f"videos/{vid_cat_name}/{vid_filename}",
                    "split": row['split'],
                    "source_identity": row['source'],
                    "target_identity": row['target1'],
                    "label": row['label'],
                    "source_video": str(row['sample_path']),
                    "frame_index": ""
                })
                
                # 2. Extract Image
                img_label_dir = "real" if row.get('video_label', row.get('v_label', 'Unknown')) == 'Real' else "fake"
                img_filename = f"{cat}_{idx}.jpg"
                dest_img_path = testing_data_dir / "images" / img_label_dir / img_filename
                extract_frame(source_path, dest_img_path)
                manifest_rows.append({
                    "modality": "image",
                    "category": img_label_dir,
                    "original_path": str(row['sample_path']),
                    "testing_path": f"images/{img_label_dir}/{img_filename}",
                    "split": row['split'],
                    "source_identity": row['source'],
                    "target_identity": row['target1'],
                    "label": "Real" if row.get('video_label', row.get('v_label', 'Unknown')) == 'Real' else "Fake",
                    "source_video": str(row['sample_path']),
                    "frame_index": "mid"
                })
                
                # 3. Extract Audio
                aud_label_dir = "real" if row.get('audio_label', row.get('a_label', 'Unknown')) == 'Real' else "fake"
                aud_filename = f"{cat}_{idx}.wav"
                dest_aud_path = testing_data_dir / "audio" / aud_label_dir / aud_filename
                extract_audio(source_path, dest_aud_path)
                manifest_rows.append({
                    "modality": "audio",
                    "category": aud_label_dir,
                    "original_path": str(row['sample_path']),
                    "testing_path": f"audio/{aud_label_dir}/{aud_filename}",
                    "split": row['split'],
                    "source_identity": row['source'],
                    "target_identity": row['target1'],
                    "label": "Real" if row.get('audio_label', row.get('a_label', 'Unknown')) == 'Real' else "Fake",
                    "source_video": str(row['sample_path']),
                    "frame_index": ""
                })
                
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(testing_data_dir / "TESTING_DATA_MANIFEST.csv", index=False)
    print("Testing data populated.")

if __name__ == "__main__":
    main()
