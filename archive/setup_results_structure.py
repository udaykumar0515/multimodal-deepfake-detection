import os
from pathlib import Path
import shutil

project_root = Path(os.path.abspath(__file__)).parent.parent

# Rename checkpoints
checkpoints_dir = project_root / "checkpoints"
for cp in checkpoints_dir.glob("v2_*.pt"):
    new_name = cp.name.replace("v2_", "")
    cp.rename(checkpoints_dir / new_name)

# Create results directories
results_dir = project_root / "results"
dirs_to_create = [
    "metrics",
    "predictions",
    "confusion_matrices",
    "roc_curves",
    "precision_recall_curves",
    "performance_curves",
    "modality_analysis",
    "error_analysis",
    "gradcam"
]
for d in dirs_to_create:
    (results_dir / d).mkdir(parents=True, exist_ok=True)

# Rename and move results
file_moves = [
    ("v2_test_metrics.json", "metrics/test_metrics.json"),
    ("v2_test_predictions.csv", "predictions/test_predictions.csv"),
    ("v2_modality_category_results.csv", "modality_analysis/modality_category_results.csv"),
    ("v2_image_confusion_matrix.png", "confusion_matrices/image_confusion_matrix.png"),
    ("v2_audio_confusion_matrix.png", "confusion_matrices/audio_confusion_matrix.png"),
    ("v2_fusion_confusion_matrix.png", "confusion_matrices/fusion_confusion_matrix.png"),
    ("v2_image_roc_curve.png", "roc_curves/image_roc_curve.png"),
    ("v2_audio_roc_curve.png", "roc_curves/audio_roc_curve.png"),
    ("v2_fusion_roc_curve.png", "roc_curves/fusion_roc_curve.png"),
    ("v2_training_history.json", "performance_curves/training_history.json")
]

for src_name, dest_rel_path in file_moves:
    src_path = results_dir / src_name
    if src_path.exists():
        src_path.rename(results_dir / dest_rel_path)

# Handle gradcam directory move if it exists
v2_gradcam_dir = results_dir / "v2_gradcam"
if v2_gradcam_dir.exists():
    for item in v2_gradcam_dir.iterdir():
        shutil.move(str(item), str(results_dir / "gradcam" / item.name))
    v2_gradcam_dir.rmdir()

# Rename scripts
scripts_dir = project_root / "scripts"
for script in scripts_dir.glob("*_v2.py"):
    new_name = script.name.replace("_v2", "")
    script.rename(scripts_dir / new_name)

print("Renaming and directory structure setup complete.")
