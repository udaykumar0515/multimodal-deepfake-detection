"""
Architecture sanity test for Phase 5.
Verifies all tensor shapes, CUDA transfer, NaN/Inf, and Grad-CAM layer accessibility.
Does NOT train the model.
"""

import os
import sys
import torch
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from dataset.dataloader_factory import create_dataloaders

def test():
    BATCH_SIZE = 2   # Deliberately small to avoid VRAM pressure during sanity test
    csv_dir   = str(project_root / "data" / "dataset_split")
    video_dir = str(project_root / "data" / "processed_frames")
    audio_dir = str(project_root / "data" / "processed_audio" / "spectrograms")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}\n")

    # ── 1. Instantiate model ──────────────────────────────────────────────────
    print("Instantiating MultimodalDeepfakeModel (pretrained=True)...")
    model = MultimodalDeepfakeModel(pretrained=True)
    model.eval()

    # ── 2. Trainable parameter count ──────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    assert trainable_params > 0, "No trainable parameters found!"

    # ── 3. Load one real batch from the existing processed dataset ────────────
    print("\nLoading one batch from the training DataLoader...")
    train_loader, val_loader, _ = create_dataloaders(
        csv_dir=csv_dir,
        video_dir=video_dir,
        audio_dir=audio_dir,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    video_batch, audio_batch, label_batch = next(iter(train_loader))

    # ── 4. Shape verification (CPU) ───────────────────────────────────────────
    print("\n--- Shape Verification (CPU) ---")
    print(f"Video input : {tuple(video_batch.shape)}  (Expected: ({BATCH_SIZE}, 16, 3, 224, 224))")
    print(f"Audio input : {tuple(audio_batch.shape)}  (Expected: ({BATCH_SIZE}, 3, 224, 224))")
    assert video_batch.shape == (BATCH_SIZE, 16, 3, 224, 224)
    assert audio_batch.shape == (BATCH_SIZE, 3, 224, 224)

    with torch.no_grad():
        # Intermediate feature shapes
        video_feat = model.video_encoder(video_batch)
        audio_feat = model.audio_encoder(audio_batch)
        fused      = torch.cat([video_feat, audio_feat], dim=1)

    print(f"Video feature: {tuple(video_feat.shape)}  (Expected: ({BATCH_SIZE}, 1280))")
    print(f"Audio feature: {tuple(audio_feat.shape)}  (Expected: ({BATCH_SIZE}, 1280))")
    print(f"Fused feature: {tuple(fused.shape)}         (Expected: ({BATCH_SIZE}, 2560))")
    assert video_feat.shape == (BATCH_SIZE, 1280)
    assert audio_feat.shape == (BATCH_SIZE, 1280)
    assert fused.shape      == (BATCH_SIZE, 2560)

    with torch.no_grad():
        logit = model(video_batch, audio_batch)

    print(f"Output logit : {tuple(logit.shape)}      (Expected: ({BATCH_SIZE}, 1))")
    assert logit.shape == (BATCH_SIZE, 1)

    # ── 5. NaN / Inf check ────────────────────────────────────────────────────
    assert not torch.isnan(logit).any(), "NaN found in model output!"
    assert not torch.isinf(logit).any(), "Inf found in model output!"
    print("NaN/Inf check: PASSED")

    # ── 6. Grad-CAM layer accessibility ──────────────────────────────────────
    print("\n--- Grad-CAM Layer Accessibility ---")
    print(f"Video grad_cam_layer type: {type(model.video_encoder.grad_cam_layer).__name__}")
    print(f"Audio grad_cam_layer type: {type(model.audio_encoder.grad_cam_layer).__name__}")
    assert model.video_encoder.grad_cam_layer is not None
    assert model.audio_encoder.grad_cam_layer is not None
    print("Grad-CAM layers accessible: PASSED")

    # ── 7. CUDA forward pass ──────────────────────────────────────────────────
    print("\n--- CUDA Test ---")
    if torch.cuda.is_available():
        model_cuda = model.to(device)
        v_cuda     = video_batch.to(device)
        a_cuda     = audio_batch.to(device)

        with torch.no_grad():
            logit_cuda = model_cuda(v_cuda, a_cuda)

        assert logit_cuda.shape == (BATCH_SIZE, 1)
        assert not torch.isnan(logit_cuda).any()
        print(f"CUDA forward pass: SUCCESS  |  output shape: {tuple(logit_cuda.shape)}")
    else:
        print("CUDA forward pass: SKIPPED (no CUDA device)")

    print("\n========================================")
    print("All architecture sanity tests PASSED.")
    print("========================================")

if __name__ == "__main__":
    test()
