"""
Plot Training Curves
====================
Reads the training_history.json file and generates plots for Loss, Accuracy, F1, AUC, and Learning Rate.
The plots are saved to the 'results' directory for research paper usage.

Usage:
    python scripts/plot_training_curves.py
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

HISTORY_FILE = project_root / "training_history.json"
RESULTS_DIR = project_root / "results"

def plot_metric(epochs, train_vals, val_vals, metric_name, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_vals, label=f'Train {metric_name}', marker='o', linewidth=2)
    if val_vals:
        plt.plot(epochs, val_vals, label=f'Validation {metric_name}', marker='s', linewidth=2)
    
    plt.title(f'Training and Validation {metric_name}', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    out_path = RESULTS_DIR / filename
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def main():
    if not HISTORY_FILE.exists():
        print(f"Error: History file not found at {HISTORY_FILE}")
        print("Please run training first to generate history.")
        sys.exit(1)
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
        
    if not history:
        print("Error: History file is empty.")
        sys.exit(1)
        
    epochs = [entry["epoch"] for entry in history]
    
    # 1. Loss
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry.get("val_loss") for entry in history]
    plot_metric(epochs, train_loss, val_loss, "Loss", "loss_curve.png")
    
    # 2. Accuracy
    train_acc = [entry.get("train_acc", 0) for entry in history]
    val_acc = [entry.get("val_acc", 0) for entry in history]
    plot_metric(epochs, train_acc, val_acc, "Accuracy", "accuracy_curve.png")
    
    # 3. F1 Score
    train_f1 = [entry.get("train_f1", 0) for entry in history]
    val_f1 = [entry.get("val_f1", 0) for entry in history]
    plot_metric(epochs, train_f1, val_f1, "F1 Score", "f1_curve.png")
    
    # 4. AUC
    train_auc = [entry.get("train_auc", 0) for entry in history]
    val_auc = [entry.get("val_auc", 0) for entry in history]
    plot_metric(epochs, train_auc, val_auc, "AUC", "auc_curve.png")
    
    # 5. Learning Rate
    lr = [entry.get("lr", 0) for entry in history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr, label='Learning Rate', marker='^', color='green', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    out_path = RESULTS_DIR / "lr_curve.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()
    
    print("\nAll plots generated successfully in the 'results' directory.")

if __name__ == "__main__":
    main()
