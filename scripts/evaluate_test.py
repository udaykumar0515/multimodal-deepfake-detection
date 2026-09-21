"""
evaluate_test.py
================
Final held-out test evaluation script for Phase 8.

Loads the best validation checkpoint and runs inference on the untouched
test set (test.csv). Calculates full metrics, saves predictions, and
generates confusion matrix and ROC curve figures.

Usage:
    python scripts/evaluate_test.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from dataset.dataloader_factory import create_dataloaders


def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==========================================")
    print(" PHASE 8: FINAL TEST EVALUATION")
    print("==========================================\n")
    print(f"Device        : {device}")
    print(f"Checkpoint    : {config['checkpoint_path']}")
    
    # ── 1. Create Loaders ──────────────────────────────────────────────────
    print("Loading test dataloader...")
    _, _, test_loader = create_dataloaders(
        csv_dir=config["csv_dir"],
        video_dir=config["video_dir"],
        audio_dir=config["audio_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )
    
    test_df = test_loader.dataset.df
    print(f"Test samples  : {len(test_df)}")
    
    # ── 2. Load Model ──────────────────────────────────────────────────────
    print("Loading model architecture...")
    model = MultimodalDeepfakeModel(pretrained=False).to(device)
    
    print("Loading checkpoint weights...")
    if not os.path.exists(config["checkpoint_path"]):
        print(f"[ERROR] Checkpoint not found: {config['checkpoint_path']}")
        sys.exit(1)
        
    ckpt = torch.load(config["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("  [OK] Checkpoint loaded successfully.")
    
    if config.get("sanity_check_only", False):
        print("\nSanity check passed. Exiting before full evaluation.")
        return

    # ── 3. Run Inference ───────────────────────────────────────────────────
    print("\nRunning inference on test set...")
    use_amp = (device.type == "cuda")
    
    all_targets = []
    all_probs = []
    
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (video, audio, labels) in enumerate(test_loader):
            video = video.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(video, audio)
            else:
                logits = model(video, audio)
                
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")
                
    total_time = time.time() - t0
    print(f"\nInference completed in {total_time:.2f} seconds.")
    
    # ── 4. Calculate Metrics ───────────────────────────────────────────────
    print("Calculating final metrics...")
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)
    preds_np = (probs_np >= 0.5).astype(int)
    
    acc = accuracy_score(targets_np, preds_np)
    prec = precision_score(targets_np, preds_np, zero_division=0)
    rec = recall_score(targets_np, preds_np, zero_division=0)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    
    try:
        auc = roc_auc_score(targets_np, probs_np)
    except ValueError:
        auc = float('nan')
        
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_np, labels=[0, 1]).ravel()
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    }
    
    print("\n[ TEST SET METRICS ]")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    
    # ── 5. Save Results ────────────────────────────────────────────────────
    results_dir = project_root / "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save Metrics JSON
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Predictions CSV
    pred_df = pd.DataFrame({
        "sample_path": test_df["sample_path"],
        "true_label": targets_np,
        "predicted_label": preds_np,
        "prob_fake": probs_np
    })
    preds_path = results_dir / "test_predictions.csv"
    pred_df.to_csv(preds_path, index=False)
    
    # Plot Confusion Matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(
        targets_np, preds_np, display_labels=["Real", "Fake"], cmap="Blues"
    )
    cm_display.ax_.set_title("Test Set Confusion Matrix")
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(targets_np, probs_np)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Test Set)')
    plt.legend(loc="lower right")
    roc_path = results_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print("\n[ ARTIFACTS SAVED ]")
    print(f"  Metrics      : {metrics_path}")
    print(f"  Predictions  : {preds_path}")
    print(f"  Conf. Matrix : {cm_path}")
    print(f"  ROC Curve    : {roc_path}")
    print("==========================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Only load model and verify checkpoint, skip full evaluation")
    args = parser.parse_args()

    config = {
        "checkpoint_path": str(project_root / "checkpoints" / "best_model.pt"),
        "csv_dir": str(project_root / "data" / "dataset_split"),
        "video_dir": str(project_root / "data" / "processed_frames"),
        "audio_dir": str(project_root / "data" / "processed_audio" / "spectrograms"),
        "batch_size": 4, # Safe for 6GB VRAM
        "num_workers": 0, # Stable on Windows
        "sanity_check_only": args.sanity_check
    }
    
    evaluate(config)
