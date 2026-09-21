"""
test_training_pipeline.py
=========================
End-to-end sanity test for Phase 6 training infrastructure.

Verifies:
    1. Focal Loss computes correctly on known inputs
    2. Forward pass produces logits
    3. Focal Loss backward pass produces gradients
    4. Optimizer updates parameters
    5. AMP (autocast + GradScaler) works correctly
    6. Validation forward + loss works without gradient flow
    7. Checkpoint save / load round-trip is correct
    8. No NaN / Inf in any output

Does NOT run a full training epoch over the dataset.
Uses a tiny number of real batches from the DataLoader.
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from dataset.dataloader_factory import create_dataloaders
from training.losses import BinaryFocalLoss
from training.trainer import Trainer

BATCH_SIZE   = 2   # Tiny for fast test
NUM_WORKERS  = 0   # Avoid multiprocessing overhead in test
N_TRAIN_BATCHES = 3
N_VAL_BATCHES   = 2


def check_no_nan(tensor: torch.Tensor, name: str) -> None:
    assert not torch.isnan(tensor).any(), f"NaN found in {name}"
    assert not torch.isinf(tensor).any(), f"Inf found in {name}"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("=" * 50)
    print(" TRAINING PIPELINE SANITY TEST")
    print("=" * 50)

    csv_dir   = str(project_root / "data" / "dataset_split")
    video_dir = str(project_root / "data" / "processed_frames")
    audio_dir = str(project_root / "data" / "processed_audio" / "spectrograms")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    print("\n[1/8] Loading DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(
        csv_dir=csv_dir, video_dir=video_dir, audio_dir=audio_dir,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )
    print(f"      Train batches available: {len(train_loader)}")
    print(f"      Val   batches available: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n[2/8] Instantiating model...")
    model = MultimodalDeepfakeModel(pretrained=False).to(device)  # pretrained=False for speed
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable parameters: {trainable:,}")
    assert trainable > 0, "No trainable parameters!"

    # ── Focal Loss ─────────────────────────────────────────────────────────────
    print("\n[3/8] Verifying Focal Loss...")
    criterion = BinaryFocalLoss(gamma=2.0)
    dummy_logits  = torch.tensor([2.0, -1.0, 0.5, -0.5])
    dummy_targets = torch.tensor([1.0,  0.0, 1.0,  0.0])
    fl_loss = criterion(dummy_logits, dummy_targets)
    assert fl_loss.item() > 0, "Focal Loss should be positive"
    check_no_nan(fl_loss, "Focal Loss")
    print(f"      Focal Loss on dummy data: {fl_loss.item():.4f}  [OK]")

    # ── Forward + Backward + Optimizer ────────────────────────────────────────
    print("\n[4/8] Forward + Backward + Optimizer update...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_amp   = (device.type == "cuda")
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    train_iter = iter(train_loader)
    params_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    for batch_idx in range(N_TRAIN_BATCHES):
        video, audio, labels = next(train_iter)
        video  = video.to(device, non_blocking=True)
        audio  = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(video, audio)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(video, audio)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        check_no_nan(logits.detach(), f"logits batch {batch_idx}")
        check_no_nan(loss.detach(),   f"loss   batch {batch_idx}")
        print(f"      Batch {batch_idx + 1}/{N_TRAIN_BATCHES}  loss={loss.item():.4f}  [OK]")

    # Verify at least some parameters changed
    params_changed = sum(
        1 for n, p in model.named_parameters()
        if p.requires_grad and not torch.allclose(p.detach(), params_before[n].to(device))
    )
    assert params_changed > 0, "No parameters were updated by the optimizer!"
    print(f"      Parameters updated: {params_changed} layers  [OK]")

    # ── Gradient check ─────────────────────────────────────────────────────────
    print("\n[5/8] Gradient check...")
    grad_count = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    assert grad_count > 0, "No gradients found after backward pass!"
    print(f"      Layers with gradients: {grad_count}  [OK]")

    # ── Validation forward ─────────────────────────────────────────────────────
    print("\n[6/8] Validation forward pass (no grad)...")
    model.eval()
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(N_VAL_BATCHES):
            video, audio, labels = next(val_iter)
            video  = video.to(device)
            audio  = audio.to(device)
            labels = labels.to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(video, audio)
                    loss   = criterion(logits, labels)
            else:
                logits = model(video, audio)
                loss   = criterion(logits, labels)
            check_no_nan(logits, "val logits")
    print(f"      Val loss on {N_VAL_BATCHES} batches: {loss.item():.4f}  [OK]")

    # ── Checkpoint save / load ─────────────────────────────────────────────────
    print("\n[7/8] Checkpoint save / load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        state = {
            "epoch":           1,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss":   0.5,
            "history":         [],
        }
        torch.save(state, ckpt_path)

        loaded = torch.load(ckpt_path, map_location=device)
        model2 = MultimodalDeepfakeModel(pretrained=False).to(device)
        model2.load_state_dict(loaded["model_state"])
        assert loaded["epoch"] == 1, "Epoch mismatch after checkpoint load"
    print("      Checkpoint round-trip: PASSED  [OK]")

    # ── Grad-CAM layers still accessible ──────────────────────────────────────
    print("\n[8/8] Grad-CAM layer accessibility after training step...")
    assert model.video_encoder.grad_cam_layer is not None
    assert model.audio_encoder.grad_cam_layer is not None
    print("      Grad-CAM layers intact  [OK]")

    print("\n" + "=" * 50)
    print(" TRAINING PIPELINE TEST PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
