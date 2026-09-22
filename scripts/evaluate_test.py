import os
import sys
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(project_root))

from models.multihead_model import MultiHeadDeepfakeModel
from dataset.multimodal_dataset import MultimodalDeepfakeDataset
from torch.utils.data import DataLoader

def plot_cm(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_roc(y_true, y_prob, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    print("Initializing Evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    test_csv = project_root / "data" / "dataset_split" / "test.csv"
    df = pd.read_csv(test_csv)
    
    dataset = MultimodalDeepfakeDataset(
        csv_path=str(test_csv),
        video_dir=str(project_root / "data" / "processed_frames"),
        audio_dir=str(project_root / "data" / "processed_audio" / "spectrograms"),
        is_train=False
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    
    print(f"Loaded checkpoint from Epoch: {checkpoint.get('epoch')}")
    
    model = MultiHeadDeepfakeModel(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_img_probs, all_aud_probs, all_fus_probs = [], [], []
    all_v_labels, all_a_labels, all_o_labels = [], [], []
    
    print("Running Inference...")
    with torch.no_grad():
        for video, audio, v_label, a_label, o_label in tqdm(loader):
            video, audio = video.to(device), audio.to(device)
            
            # Use torch.amp.autocast
            with torch.amp.autocast('cuda'):
                preds = model(video=video, audio=audio, return_all=True)
                
            img_logits = preds['image'] # (B*16, 1)
            img_probs = torch.sigmoid(img_logits).view(video.size(0), 16).mean(dim=1) # (B)
            
            aud_probs = torch.sigmoid(preds['audio']).view(-1)
            fus_probs = torch.sigmoid(preds['fusion']).view(-1)
            
            all_img_probs.extend(img_probs.cpu().numpy())
            all_aud_probs.extend(aud_probs.cpu().numpy())
            all_fus_probs.extend(fus_probs.cpu().numpy())
            
            all_v_labels.extend(v_label.view(-1).numpy())
            all_a_labels.extend(a_label.view(-1).numpy())
            all_o_labels.extend(o_label.view(-1).numpy())
            
    df['img_prob'] = all_img_probs
    df['img_pred'] = (df['img_prob'] > 0.5).astype(float)
    df['aud_prob'] = all_aud_probs
    df['aud_pred'] = (df['aud_prob'] > 0.5).astype(float)
    df['fus_prob'] = all_fus_probs
    df['fus_pred'] = (df['fus_prob'] > 0.5).astype(float)
    
    df['v_label_num'] = all_v_labels
    df['a_label_num'] = all_a_labels
    df['o_label_num'] = all_o_labels
    
    # Calculate metrics
    def get_metrics(y_true, y_prob, y_pred):
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_true, y_prob))
        }
        
    img_metrics = get_metrics(df['v_label_num'], df['img_prob'], df['img_pred'])
    aud_metrics = get_metrics(df['a_label_num'], df['aud_prob'], df['aud_pred'])
    fus_metrics = get_metrics(df['o_label_num'], df['fus_prob'], df['fus_pred'])
    
    # 4-Category
    categories = ["RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]
    cat_results = []
    
    for cat in categories:
        cat_df = df[df['type'] == cat]
        n = len(cat_df)
        if n == 0: continue
        
        img_acc = accuracy_score(cat_df['v_label_num'], cat_df['img_pred'])
        aud_acc = accuracy_score(cat_df['a_label_num'], cat_df['aud_pred'])
        fus_acc = accuracy_score(cat_df['o_label_num'], cat_df['fus_pred'])
        
        cat_results.append({
            "Category": cat,
            "N": n,
            "Expected Video Label": "Fake" if cat_df['v_label_num'].iloc[0] == 1 else "Real",
            "Expected Audio Label": "Fake" if cat_df['a_label_num'].iloc[0] == 1 else "Real",
            "Expected Overall Label": "Fake" if cat_df['o_label_num'].iloc[0] == 1 else "Real",
            "Image Acc": img_acc,
            "Audio Acc": aud_acc,
            "Fusion Acc": fus_acc,
            "Image Mean Fake Prob": float(cat_df['img_prob'].mean()),
            "Audio Mean Fake Prob": float(cat_df['aud_prob'].mean()),
            "Fusion Mean Fake Prob": float(cat_df['fus_prob'].mean())
        })
        
    cat_df_final = pd.DataFrame(cat_results)
    cat_df_final.to_csv(results_dir / "modality_analysis" / "modality_category_results.csv", index=False)
    
    # Save Metrics JSON
    final_metrics = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_sample_count": len(df),
        "image_metrics": img_metrics,
        "audio_metrics": aud_metrics,
        "fusion_metrics": fus_metrics
    }
    with open(results_dir / "metrics" / "test_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Save predictions
    df.to_csv(results_dir / "predictions" / "test_predictions.csv", index=False)
    
    # 5. Visualizations
    plot_cm(df['v_label_num'], df['img_pred'], "Image Head Confusion Matrix", results_dir / "confusion_matrices" / "image_confusion_matrix.png")
    plot_cm(df['a_label_num'], df['aud_pred'], "Audio Head Confusion Matrix", results_dir / "confusion_matrices" / "audio_confusion_matrix.png")
    plot_cm(df['o_label_num'], df['fus_pred'], "Fusion Head Confusion Matrix", results_dir / "confusion_matrices" / "fusion_confusion_matrix.png")
    
    plot_roc(df['v_label_num'], df['img_prob'], "Image Head ROC Curve", results_dir / "roc_curves" / "image_roc_curve.png")
    plot_roc(df['a_label_num'], df['aud_prob'], "Audio Head ROC Curve", results_dir / "roc_curves" / "audio_roc_curve.png")
    plot_roc(df['o_label_num'], df['fus_prob'], "Fusion Head ROC Curve", results_dir / "roc_curves" / "fusion_roc_curve.png")

    print("\nDONE! Saved all outputs to results/")

if __name__ == "__main__":
    main()
