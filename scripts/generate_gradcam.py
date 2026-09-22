import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.multihead_model import MultiHeadDeepfakeModel
from dataset.multimodal_dataset import MultimodalDeepfakeDataset
from torch.utils.data import DataLoader

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, video=None, audio=None, target_head='image'):
        """
        target_head: 'image', 'audio', or 'fusion'
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        preds = self.model(video=video, audio=audio, return_all=True)
        target_logits = preds[target_head] # (B, 1) or (B*T, 1) for image
        
        # Calculate loss to backpropagate
        # We want to explain the "Fake" class, so we take the sigmoid and backprop
        prob = torch.sigmoid(target_logits).mean() # Take mean if multiple frames
        prob.backward()
        
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        # Global Average Pooling on gradients
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * activations, axis=1, keepdims=True)
        cam = np.maximum(cam, 0) # ReLU
        
        # Normalize
        cam_min = cam.min(axis=(2, 3), keepdims=True)
        cam_max = cam.max(axis=(2, 3), keepdims=True)
        heatmap = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return heatmap, prob.item()

def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = tensor.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def create_overlay(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_color, 0.5, 0)
    return overlay

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Output directory
    output_dir = project_root / "results" / "gradcam"
    subdirs = ['RealVideo-RealAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio', 'FakeVideo-FakeAudio']
    for s in subdirs:
        (output_dir / s).mkdir(parents=True, exist_ok=True)
        
    # 2. Dataset
    test_csv = project_root / "data" / "dataset_split" / "test.csv"
    df = pd.read_csv(test_csv)
    dataset = MultimodalDeepfakeDataset(
        csv_path=str(test_csv),
        video_dir=str(project_root / "data" / "processed_frames"),
        audio_dir=str(project_root / "data" / "processed_audio" / "spectrograms"),
        is_train=False
    )
    
    # 3. Model
    model = MultiHeadDeepfakeModel(pretrained=False).to(device)
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from Epoch {checkpoint['epoch']}")
    
    # Initialize GradCAM tools
    visual_gcam = GradCAM(model, model.visual_encoder.grad_cam_layer)
    audio_gcam = GradCAM(model, model.audio_encoder.grad_cam_layer)
    
    # Select 1 sample per category
    selected_indices = []
    for cat in subdirs:
        cat_df = df[df['type'] == cat]
        if len(cat_df) > 0:
            idx = cat_df.index[0] # Just take the first one
            selected_indices.append((idx, cat))
            
    for idx, cat in selected_indices:
        print(f"\nProcessing {cat}...")
        row = df.iloc[idx]
        sample_path = row['sample_path']
        v_label_true = row['v_label'] if 'v_label' in row else row.get('video_label', 'Unknown')
        a_label_true = row['a_label'] if 'a_label' in row else row.get('audio_label', 'Unknown')
        o_label_true = row['label'] if 'label' in row else row.get('overall_label', 'Unknown')
        
        video, audio, v_lbl, a_lbl, o_lbl = dataset[idx]
        # Expand batch dim
        video = video.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)
        
        # ----------- VISUAL GRAD-CAM -----------
        # We need image grad-cam on Image Head
        video.requires_grad_(True)
        vis_heatmap, vis_prob = visual_gcam(video=video, audio=audio, target_head='image')
        # vis_heatmap is shape (B*16, 1, H_f, W_f)
        # We need frames 0, 7, 15
        target_frames = [0, 7, 15]
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        fig.suptitle(f"{cat}\nTrue Overall: {o_label_true} | Image Head Pred Prob (Fake): {vis_prob:.4f}")
        
        video_np = video.squeeze(0).cpu().detach().numpy() # (16, 3, 224, 224)
        vis_heatmap = vis_heatmap[:, 0, :, :] # (16, H_f, W_f)
        
        for i, f_idx in enumerate(target_frames):
            orig_img = denormalize_image(video_np[f_idx])
            hm = vis_heatmap[f_idx]
            
            overlay = create_overlay(orig_img, hm)
            
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Frame {f_idx} (Original)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"Frame {f_idx} (Grad-CAM)")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        vis_out_path = output_dir / cat / "visual_gradcam.png"
        plt.savefig(vis_out_path)
        plt.close()
        
        # ----------- AUDIO GRAD-CAM -----------
        model.zero_grad()
        audio.requires_grad_(True)
        aud_heatmap, aud_prob = audio_gcam(video=video, audio=audio, target_head='audio')
        # aud_heatmap is (1, 1, H_f, W_f)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{cat}\nTrue Audio Label: {a_label_true} | Audio Head Pred Prob (Fake): {aud_prob:.4f}")
        
        aud_np = audio.squeeze(0).cpu().detach().numpy() # (3, 224, 224)
        orig_aud = denormalize_image(aud_np)
        
        hm_aud = aud_heatmap[0, 0]
        aud_overlay = create_overlay(orig_aud, hm_aud)
        
        axes[0].imshow(orig_aud)
        axes[0].set_title("Audio Spectrogram (Original)")
        axes[0].axis('off')
        
        axes[1].imshow(aud_overlay)
        axes[1].set_title("Audio Spectrogram (Grad-CAM)")
        axes[1].axis('off')
        
        plt.tight_layout()
        aud_out_path = output_dir / cat / "audio_gradcam.png"
        plt.savefig(aud_out_path)
        plt.close()
        
        # Also get fusion probability for logging
        with torch.no_grad():
            preds = model(video=video, audio=audio, return_all=True)
            fus_prob = torch.sigmoid(preds['fusion']).item()
            
        # Write metadata
        meta = f"Sample: {sample_path}\n"
        meta += f"Category: {cat}\n"
        meta += f"True Labels -> Video: {v_label_true}, Audio: {a_label_true}, Overall: {o_label_true}\n"
        meta += f"V2 Predictions (Fake Prob) -> Image Head: {vis_prob:.4f} | Audio Head: {aud_prob:.4f} | Fusion Head: {fus_prob:.4f}\n"
        
        with open(output_dir / cat / "metadata.txt", "w") as f:
            f.write(meta)
            
        print(f"Saved {cat} outputs.")

if __name__ == "__main__":
    main()
