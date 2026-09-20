import cv2
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import os

try:
    from insightface.app import FaceAnalysis  # type: ignore
    HAVE_INSIGHTFACE = True
except ImportError:
    HAVE_INSIGHTFACE = False
    print("Warning: insightface not installed. RetinaFace will not work.")

try:
    import albumentations as A  # type: ignore
    from albumentations.pytorch import ToTensorV2  # type: ignore
    HAVE_ALBUMENTATIONS = True
except ImportError:
    HAVE_ALBUMENTATIONS = False


class UniformTemporalSampler:
    """Samples frames uniformly from a video."""
    def __init__(self, num_frames: int = 16):
        self.num_frames = num_frames

    def sample(self, video_path: str) -> List[np.ndarray]:
        """Samples frames from video deterministically."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video {video_path} has 0 frames.")

        if total_frames < self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        current_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_idx in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                counts = np.count_nonzero(indices == current_idx)
                for _ in range(counts):
                    frames.append(frame_rgb)
                    
            current_idx += 1
            if len(frames) >= self.num_frames:
                break
                
        cap.release()
        
        while len(frames) < self.num_frames and len(frames) > 0:
            frames.append(frames[-1])
            
        if not frames:
            raise ValueError(f"Failed to read any frames from {video_path}")
            
        return frames

class RetinaFaceCropper:
    """Detects faces using RetinaFace and applies a 20% margin."""
    def __init__(self, margin: float = 0.20, target_size: Tuple[int, int] = (224, 224)):
        self.margin = margin
        self.target_size = target_size
        
        if HAVE_INSIGHTFACE:
            self.app = FaceAnalysis(allowed_modules=['detection'], providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def crop_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Finds face, expands bbox, and crops."""
        h, w, _ = frame.shape
        
        if HAVE_INSIGHTFACE:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            faces = self.app.get(frame_bgr)
            
            if len(faces) == 0:
                return None, {"reason": "no_face_detected"}
                
            largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            x1, y1, x2, y2 = map(int, largest_face.bbox)
        else:
            x1, y1, x2, y2 = w//4, h//4, w*3//4, h*3//4
            
        box_w = x2 - x1
        box_h = y2 - y1
        
        margin_w = int(box_w * self.margin)
        margin_h = int(box_h * self.margin)
        
        new_x1 = max(0, x1 - margin_w)
        new_y1 = max(0, y1 - margin_h)
        new_x2 = min(w, x2 + margin_w)
        new_y2 = min(h, y2 + margin_h)
        
        crop = frame[new_y1:new_y2, new_x1:new_x2]
        
        if crop.size == 0:
             return None, {"reason": "invalid_crop_size"}
             
        crop_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        return crop_resized, {"bbox": (new_x1, new_y1, new_x2, new_y2), "original_bbox": (x1, y1, x2, y2)}

class VisualTransform:
    """Applies visual augmentations and normalization."""
    def __init__(self, is_train: bool = False):
        self.is_train = is_train
        
        if HAVE_ALBUMENTATIONS:
            normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            if is_train:
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, p=1.0, interpolation=cv2.INTER_CUBIC),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0),
                    A.CoarseDropout(max_holes=1, max_height=int(224*0.1), max_width=int(224*0.1), 
                                   min_holes=1, min_height=int(224*0.02), min_width=int(224*0.02), p=0.1),
                    normalize,
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    normalize,
                    ToTensorV2()
                ])
        else:
            self.transform = None

    def apply(self, image: np.ndarray) -> torch.Tensor:
        if self.transform:
            return self.transform(image=image)["image"]
        else:
            img = image.astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)

class VideoPreprocessor:
    """End-to-end video preprocessing pipeline."""
    def __init__(self, num_frames: int = 16, margin: float = 0.20, is_train: bool = False):
        self.sampler = UniformTemporalSampler(num_frames=num_frames)
        self.cropper = RetinaFaceCropper(margin=margin)
        self.transform = VisualTransform(is_train=is_train)
        
    def process(self, video_path: str) -> Tuple[Optional[torch.Tensor], List[dict]]:
        """Processes video into (16, 3, 224, 224) tensor."""
        try:
            frames = self.sampler.sample(video_path)
        except Exception as e:
            return None, [{"reason": f"sampling_failed: {str(e)}"}]
            
        processed_frames = []
        failures = []
        
        for idx, frame in enumerate(frames):
            crop, info = self.cropper.crop_face(frame)
            if crop is None:
                failures.append({"frame_idx": idx, "reason": info["reason"]})
                h, w, _ = frame.shape
                cw, ch = 224, 224
                x1, y1 = max(0, w//2 - cw//2), max(0, h//2 - ch//2)
                x2, y2 = min(w, x1 + cw), min(h, y1 + ch)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (224, 224))
                else:
                    crop = np.zeros((224, 224, 3), dtype=np.uint8)
                    
            tensor = self.transform.apply(crop)
            processed_frames.append(tensor)
            
        final_tensor = torch.stack(processed_frames)
        return final_tensor, failures
