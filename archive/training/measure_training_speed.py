"""
Phase 6.5 — Training Readiness / Hardware Timing Check
=======================================================
Measures real training throughput, VRAM usage, and produces time estimates.
Does NOT perform full training or save persistent checkpoints.
"""

import os
import sys
import time
import tempfile
import statistics
import torch
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from dataset.dataloader_factory import create_dataloaders
from training.losses import BinaryFocalLoss

BATCH_SIZE      = 4
WARMUP_BATCHES  = 5      # Discarded from timing
MEASURE_BATCHES = 30     # Measured for stable average
VAL_BATCHES     = 20

TRAIN_TOTAL = 15083
VAL_TOTAL   = 3191


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    use_amp  = (device.type == "cuda")

    print("\n==========================================")
    print(" TRAINING READINESS REPORT")
    print("==========================================\n")
    print(f"GPU     : {gpu_name}")
    print(f"Device  : {device}")
    print(f"AMP     : {'enabled' if use_amp else 'disabled'}")
    print(f"Batch   : {BATCH_SIZE}")
    print(f"Warm-up : {WARMUP_BATCHES} batches (discarded)")
    print(f"Measure : {MEASURE_BATCHES} batches\n")

    # ── DataLoader ────────────────────────────────────────────────────────────
    csv_dir   = str(project_root / "data" / "dataset_split")
    video_dir = str(project_root / "data" / "processed_frames")
    audio_dir = str(project_root / "data" / "processed_audio" / "spectrograms")

    train_loader, val_loader, _ = create_dataloaders(
        csv_dir=csv_dir, video_dir=video_dir, audio_dir=audio_dir,
        batch_size=BATCH_SIZE, num_workers=0,  # 0 avoids worker crashes on Windows
    )

    # ── Model & Loss ──────────────────────────────────────────────────────────
    model     = MultimodalDeepfakeModel(pretrained=True).to(device)
    criterion = BinaryFocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    # Reset CUDA memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── Training timing ───────────────────────────────────────────────────────
    model.train()
    train_iter  = iter(train_loader)
    batch_times = []
    oom_hit     = False
    nan_hit     = False

    print("Running training batches...")
    for i in range(WARMUP_BATCHES + MEASURE_BATCHES):
        try:
            video, audio, labels = next(train_iter)
        except StopIteration:
            break

        video  = video.to(device, non_blocking=True)
        audio  = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        try:
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

        except torch.cuda.OutOfMemoryError:
            oom_hit = True
            print("  [OOM] CUDA out of memory!")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if torch.isnan(loss) or torch.isinf(loss):
            nan_hit = True

        if i >= WARMUP_BATCHES:
            batch_times.append(elapsed)
            print(f"  Batch {i - WARMUP_BATCHES + 1:02d}/{MEASURE_BATCHES}  "
                  f"loss={loss.item():.4f}  t={elapsed:.3f}s")

    # ── VRAM ──────────────────────────────────────────────────────────────────
    peak_alloc   = 0.0
    peak_reserve = 0.0
    if device.type == "cuda":
        peak_alloc   = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserve = torch.cuda.max_memory_reserved(device)  / 1024**3

    # ── Validation timing ─────────────────────────────────────────────────────
    print("\nRunning validation batches...")
    model.eval()
    val_iter   = iter(val_loader)
    val_times  = []

    with torch.no_grad():
        for i in range(VAL_BATCHES):
            try:
                video, audio, labels = next(val_iter)
            except StopIteration:
                break
            video  = video.to(device, non_blocking=True)
            audio  = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(video, audio)
                    loss   = criterion(logits, labels)
            else:
                logits = model(video, audio)
                loss   = criterion(logits, labels)

            if device.type == "cuda":
                torch.cuda.synchronize()
            val_times.append(time.perf_counter() - t0)

    # ── Checkpoint save test ───────────────────────────────────────────────────
    ckpt_ok = False
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            ckpt_path = os.path.join(tmpdir, "readiness_test.pt")
            torch.save({
                "epoch": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            loaded = torch.load(ckpt_path, map_location=device)
            assert "model_state" in loaded
            ckpt_ok = True
        except Exception as e:
            print(f"  Checkpoint save failed: {e}")

    # ── Calculations ──────────────────────────────────────────────────────────
    avg_train  = statistics.mean(batch_times)  if batch_times else float("nan")
    med_train  = statistics.median(batch_times) if batch_times else float("nan")
    avg_val    = statistics.mean(val_times)    if val_times    else float("nan")

    train_batches_per_epoch = -(-TRAIN_TOTAL // BATCH_SIZE)  # ceiling division
    val_batches_per_epoch   = -(-VAL_TOTAL   // BATCH_SIZE)

    train_epoch_sec = avg_train * train_batches_per_epoch
    val_epoch_sec   = avg_val   * val_batches_per_epoch
    epoch_total_sec = train_epoch_sec + val_epoch_sec

    videos_per_sec = BATCH_SIZE / avg_train if avg_train > 0 else 0

    est_5  = epoch_total_sec * 5
    est_10 = epoch_total_sec * 10
    est_15 = epoch_total_sec * 15

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n==========================================")
    print(" TRAINING READINESS REPORT — RESULTS")
    print("==========================================")
    print(f"\nGPU                 : {gpu_name}")
    print(f"CUDA                : {torch.version.cuda}")
    print(f"AMP                 : {'enabled' if use_amp else 'disabled'}")
    print(f"Batch size tested   : {BATCH_SIZE}")
    print(f"Training batches    : {len(batch_times)} measured (after {WARMUP_BATCHES} warm-up)")
    print(f"Avg batch time      : {avg_train:.3f}s")
    print(f"Median batch time   : {med_train:.3f}s")
    print(f"Videos/sec          : {videos_per_sec:.2f}")
    print(f"Peak alloc VRAM     : {peak_alloc:.2f} GB")
    print(f"Peak reserved VRAM  : {peak_reserve:.2f} GB")
    print(f"\nVal batches measured: {len(val_times)}")
    print(f"Avg val batch time  : {avg_val:.3f}s")
    print(f"\nEst. train epoch    : {fmt_time(train_epoch_sec)}")
    print(f"Est. val epoch      : {fmt_time(val_epoch_sec)}")
    print(f"Est. total/epoch    : {fmt_time(epoch_total_sec)}")
    print(f"\nEst.  5 epochs      : {fmt_time(est_5)}")
    print(f"Est. 10 epochs      : {fmt_time(est_10)}")
    print(f"Est. 15 epochs      : {fmt_time(est_15)}")
    print(f"\nCUDA OOM            : {'YES' if oom_hit else 'No'}")
    print(f"NaN/Inf             : {'YES' if nan_hit else 'No'}")
    print(f"Checkpoint save     : {'PASSED' if ckpt_ok else 'FAILED'}")
    print(f"\nOverall status      : {'READY' if not oom_hit and not nan_hit and ckpt_ok else 'ISSUES DETECTED'}")
    print("==========================================\n")

    # Return key stats for PROJECT_LOG update
    return {
        "gpu": gpu_name,
        "batch_size": BATCH_SIZE,
        "avg_batch_sec": round(avg_train, 3),
        "videos_per_sec": round(videos_per_sec, 2),
        "peak_alloc_gb": round(peak_alloc, 2),
        "peak_reserve_gb": round(peak_reserve, 2),
        "train_epoch_sec": round(train_epoch_sec),
        "val_epoch_sec": round(val_epoch_sec),
        "epoch_total_sec": round(epoch_total_sec),
        "est_5": round(est_5),
        "est_10": round(est_10),
        "est_15": round(est_15),
        "oom": oom_hit,
        "nan": nan_hit,
        "ckpt_ok": ckpt_ok,
    }


if __name__ == "__main__":
    main()
