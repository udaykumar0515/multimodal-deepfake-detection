"""
Phase 9A: Grad-CAM Preparation and Sanity Check
===============================================
Extracts one single sample from the test set.
Validates if `model.video_encoder.grad_cam_layer` and 
`model.audio_encoder.grad_cam_layer` can successfully produce 
activations and gradients via backward hooks.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from dataset.dataloader_factory import create_dataloaders


class DummyGradCAM:
    """Minimal logic to capture gradients and activations."""
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
        # grad_output[0] is the gradient w.r.t the layer's output
        self.gradients = grad_output[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==========================================")
    print(" PHASE 9A: GRAD-CAM SANITY CHECK")
    print("==========================================\n")
    
    # ── 1. Load Model & Checkpoint ──────────────────────────────────────────
    ckpt_path = str(project_root / "checkpoints" / "best_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    
    model = MultimodalDeepfakeModel(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # Eval mode for inference, but we still need gradients
    print("  [OK] Checkpoint loaded.")

    # ── 2. Get One Test Sample ──────────────────────────────────────────────
    print("\nExtracting one test sample...")
    _, _, test_loader = create_dataloaders(
        csv_dir=str(project_root / "data" / "dataset_split"),
        video_dir=str(project_root / "data" / "processed_frames"),
        audio_dir=str(project_root / "data" / "processed_audio" / "spectrograms"),
        batch_size=1, 
        num_workers=0
    )
    
    test_iter = iter(test_loader)
    video, audio, labels = next(test_iter)
    video = video.to(device)
    audio = audio.to(device)
    label = labels.to(device)
    
    sample_path = test_loader.dataset.df.iloc[0]['sample_path']
    print(f"  [OK] Sample loaded: {sample_path}")
    print(f"  [OK] True label: {label.item()}")

    # ── 3. Attach Hooks ─────────────────────────────────────────────────────
    print("\nAttaching Grad-CAM hooks...")
    
    # Enable gradient tracking on input for the backward pass to reach the weights
    for param in model.parameters():
        param.requires_grad = True
        
    cam_video = DummyGradCAM(model, model.video_encoder.grad_cam_layer)
    cam_audio = DummyGradCAM(model, model.audio_encoder.grad_cam_layer)
    print("  [OK] Hooks registered.")

    # ── 4. Forward Pass ─────────────────────────────────────────────────────
    print("\nRunning forward pass...")
    model.zero_grad()
    logits = model(video, audio)
    prob = torch.sigmoid(logits).item()
    print(f"  [OK] Prediction (prob_fake): {prob:.4f}")

    # ── 5. Backward Pass ────────────────────────────────────────────────────
    print("\nRunning backward pass to compute gradients...")
    # For Grad-CAM, we backpropagate from the logit corresponding to the class of interest.
    # We'll use the raw logit (representing the 'Fake' class directly).
    logits.backward(retain_graph=True)
    print("  [OK] Backward pass completed.")

    # ── 6. Verify Video Grad-CAM ────────────────────────────────────────────
    print("\n[ Video Encoder Grad-CAM Verification ]")
    v_act = cam_video.activations
    v_grad = cam_video.gradients
    
    if v_act is not None and v_grad is not None:
        print(f"  [OK] Activations captured: {v_act.shape}")
        print(f"  [OK] Gradients captured  : {v_grad.shape}")
        if v_act.shape == v_grad.shape:
            print("  [OK] Shapes match.")
            print("  [OK] Video Grad-CAM is TECHNICALLY SUPPORTED.")
        else:
            print("  [ERROR] Shape mismatch between video activations and gradients.")
    else:
        print("  [ERROR] Failed to capture video activations/gradients.")

    # ── 7. Verify Audio Grad-CAM ────────────────────────────────────────────
    print("\n[ Audio Encoder Grad-CAM Verification ]")
    a_act = cam_audio.activations
    a_grad = cam_audio.gradients
    
    if a_act is not None and a_grad is not None:
        print(f"  [OK] Activations captured: {a_act.shape}")
        print(f"  [OK] Gradients captured  : {a_grad.shape}")
        if a_act.shape == a_grad.shape:
            print("  [OK] Shapes match.")
            print("  [OK] Audio Grad-CAM is TECHNICALLY SUPPORTED.")
        else:
            print("  [ERROR] Shape mismatch between audio activations and gradients.")
    else:
        print("  [ERROR] Failed to capture audio activations/gradients.")
        
    print("\n==========================================")
    print(" SANITY CHECK COMPLETE")
    print("==========================================\n")


if __name__ == "__main__":
    main()
