import os
import sys
import cv2
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from dataset.dataloader_factory import create_dataloaders


class GradCAM:
    """Reusable Grad-CAM implementation for a specific target layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self):
        """Generates heatmap from captured gradients and activations."""
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured.")
            
        # Global Average Pooling of gradients (B, C, H, W) -> (B, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1)
        
        # ReLU to keep only features that have a positive influence on the class
        cam = F.relu(cam)
        
        # Normalize to [0, 1] across spatial dimensions per batch item
        cam = cam.cpu().numpy()
        cam_min = cam.min(axis=(1,2), keepdims=True)
        cam_max = cam.max(axis=(1,2), keepdims=True)
        # Avoid division by zero
        heatmap = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return heatmap


def denormalize_image(tensor):
    """Reverts ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = tensor.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def create_overlay(original_img, heatmap):
    """Creates a colored heatmap overlay on the original image."""
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    # Convert heatmap to uint8 color
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Overlay with original image
    # Note: OpenCV expects BGR. Our original_img is RGB. We convert heatmap to RGB for blending.
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_color, 0.5, 0)
    return heatmap_resized, overlay


def save_visualizations(out_dir, prefix, orig, heatmap_resized, overlay):
    """Saves the visual artifacts."""
    # Convert RGB to BGR for OpenCV saving
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    
    # Apply colormap to heatmap just for standalone saving
    hm_bgr = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    cv2.imwrite(f"{out_dir}/{prefix}_original.png", orig_bgr)
    cv2.imwrite(f"{out_dir}/{prefix}_heatmap.png", hm_bgr)
    cv2.imwrite(f"{out_dir}/{prefix}_overlay.png", overlay_bgr)


def run_gradcam(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── 1. Setup Directories ──
    out_root = project_root / "results" / "gradcam"
    dirs = {
        "correct_real": out_root / "correct_real",
        "correct_fake": out_root / "correct_fake",
        "false_negatives": out_root / "false_negatives"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    metadata = []
    
    # ── 2. Target Samples ──
    # The false negatives explicitly requested
    target_fns = [
        "FakeVideo-RealAudio/African/women/id00460/00005.mp4",
        "FakeVideo-RealAudio/African/women/id00592/00017.mp4",
        "FakeVideo-RealAudio/Asian (East)/men/id06591/00021.mp4"
    ]
    
    # We will pick the first correct real and first correct fake we encounter
    found_correct_real = False
    found_correct_fake = False
    
    # ── 3. Load Model ──
    model = MultimodalDeepfakeModel(pretrained=False).to(device)
    ckpt_path = project_root / "checkpoints" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Attach Grad-CAM
    for param in model.parameters():
        param.requires_grad = True
        
    cam_video = GradCAM(model, model.video_encoder.grad_cam_layer)
    cam_audio = GradCAM(model, model.audio_encoder.grad_cam_layer)
    
    # ── 4. DataLoader ──
    _, _, test_loader = create_dataloaders(
        csv_dir=str(project_root / "data" / "dataset_split"),
        video_dir=str(project_root / "data" / "processed_frames"),
        audio_dir=str(project_root / "data" / "processed_audio" / "spectrograms"),
        batch_size=1, 
        num_workers=0
    )
    
    # Selected deterministic frames to visualize
    target_frames = [0, 7, 15]
    
    processed_count = 0
    fn_processed = 0
    
    print("Beginning Grad-CAM generation...")
    
    for idx, (video, audio, label) in enumerate(test_loader):
        sample_path = test_loader.dataset.df.iloc[idx]['sample_path']
        true_label = int(label.item())
        
        is_target_fn = sample_path in target_fns
        
        # Decide if we need this sample
        category = None
        if is_target_fn:
            category = "false_negatives"
            prefix = f"fn_{fn_processed+1:02d}"
        elif true_label == 0 and not found_correct_real:
            # Check if prediction is actually correct real
            # Wait, we need to run inference first to know
            category = "correct_real"
            prefix = "real_01"
        elif true_label == 1 and not found_correct_fake:
            category = "correct_fake"
            prefix = "fake_01"
        else:
            continue
            
        # Inference
        video_dev = video.to(device)
        audio_dev = audio.to(device)
        
        model.zero_grad()
        logits = model(video_dev, audio_dev)
        prob_fake = torch.sigmoid(logits).item()
        pred_label = 1 if prob_fake >= 0.5 else 0
        
        # Verify conditions
        if category == "correct_real" and pred_label != 0:
            continue  # Was not correct, skip
        if category == "correct_fake" and pred_label != 1:
            continue
            
        if category == "correct_real": found_correct_real = True
        if category == "correct_fake": found_correct_fake = True
        if category == "false_negatives": fn_processed += 1
        
        print(f"Processing [{category}]: {sample_path} (Prob Fake: {prob_fake:.4f})")
        
        # Backward pass
        # Backprop from the logit corresponding to the fake class
        logits.backward(retain_graph=True)
        
        # Generate Heatmaps
        hm_video = cam_video.generate_heatmap() # (16, 7, 7)
        hm_audio = cam_audio.generate_heatmap() # (1, 7, 7)
        
        out_dir = str(dirs[category])
        
        # Video Processing (Frames)
        for f_idx in target_frames:
            orig_frame = video[0, f_idx].numpy() # (3, 224, 224)
            orig_frame_img = denormalize_image(orig_frame)
            hm_f = hm_video[f_idx]
            
            hm_resized, overlay = create_overlay(orig_frame_img, hm_f)
            
            f_prefix = f"{prefix}_frame_{f_idx:02d}"
            save_visualizations(out_dir, f_prefix, orig_frame_img, hm_resized, overlay)
            
            metadata.append({
                "sample_path": sample_path,
                "true_label": true_label,
                "predicted_label": pred_label,
                "fake_probability": round(prob_fake, 5),
                "modality": "video",
                "frame_index": f_idx,
                "artifact_prefix": f"{category}/{f_prefix}"
            })
            
        # Audio Processing
        orig_audio = audio[0].numpy() # (3, 224, 224)
        orig_audio_img = denormalize_image(orig_audio)
        hm_a = hm_audio[0]
        
        hm_resized_a, overlay_a = create_overlay(orig_audio_img, hm_a)
        
        a_prefix = f"{prefix}_audio"
        save_visualizations(out_dir, a_prefix, orig_audio_img, hm_resized_a, overlay_a)
        
        metadata.append({
            "sample_path": sample_path,
            "true_label": true_label,
            "predicted_label": pred_label,
            "fake_probability": round(prob_fake, 5),
            "modality": "audio",
            "frame_index": None,
            "artifact_prefix": f"{category}/{a_prefix}"
        })
        
        processed_count += 1
        
        # Stop condition
        if found_correct_real and found_correct_fake and fn_processed == 3:
            break
            
    # ── 5. Save Metadata ──
    meta_path = out_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nCompleted processing {processed_count} samples.")
    print(f"Results saved to: {out_root}")

if __name__ == "__main__":
    config = {}
    run_gradcam(config)
