import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path

project_root = Path(os.path.abspath(__file__)).parent.parent
results_dir = project_root / "results"

# Read test predictions
df_path = results_dir / "predictions" / "test_predictions.csv"
if not df_path.exists():
    print("test_predictions.csv not found")
    exit(1)
df = pd.read_csv(df_path)

# 1. Precision-Recall Curves
pr_dir = results_dir / "precision_recall_curves"
pr_dir.mkdir(parents=True, exist_ok=True)

def plot_pr(y_true, y_prob, title, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

plot_pr(df['v_label_num'], df['img_prob'], "Image Head PR Curve", pr_dir / "image_pr_curve.png")
plot_pr(df['a_label_num'], df['aud_prob'], "Audio Head PR Curve", pr_dir / "audio_pr_curve.png")
plot_pr(df['o_label_num'], df['fus_prob'], "Fusion Head PR Curve", pr_dir / "fusion_pr_curve.png")

# 2. Error Analysis
error_dir = results_dir / "error_analysis"
error_dir.mkdir(parents=True, exist_ok=True)

# Define errors for fusion head as the primary system
fp = df[(df['o_label_num'] == 0) & (df['fus_pred'] == 1)]
fn = df[(df['o_label_num'] == 1) & (df['fus_pred'] == 0)]

fp.to_csv(error_dir / "false_positives.csv", index=False)
fn.to_csv(error_dir / "false_negatives.csv", index=False)

summary = {
    "Total Samples": len(df),
    "False Positives": len(fp),
    "False Negatives": len(fn),
    "Image Head Errors": len(df[df['v_label_num'] != df['img_pred']]),
    "Audio Head Errors": len(df[df['a_label_num'] != df['aud_pred']]),
    "Fusion Head Errors": len(df[df['o_label_num'] != df['fus_pred']])
}
pd.DataFrame([summary]).to_csv(error_dir / "error_summary.csv", index=False)

# 3. Dataset Summary
dataset_summary = {
    "Total Test Samples": len(df),
    "RealVideo-RealAudio": len(df[df['type'] == 'RealVideo-RealAudio']),
    "FakeVideo-RealAudio": len(df[df['type'] == 'FakeVideo-RealAudio']),
    "RealVideo-FakeAudio": len(df[df['type'] == 'RealVideo-FakeAudio']),
    "FakeVideo-FakeAudio": len(df[df['type'] == 'FakeVideo-FakeAudio'])
}
with open(results_dir / "dataset_summary.json", "w") as f:
    json.dump(dataset_summary, f, indent=4)

print("Generated PR Curves, Error Analysis, and Dataset Summary.")
