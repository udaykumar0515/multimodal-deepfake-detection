# Deepfake Detection Using Multimodal Learning

## Overview

This project implements a Multi-Task Deepfake Detection architecture capable of independently analyzing visual and audio streams to detect synthetic manipulation. As generative AI becomes increasingly sophisticated, unimodal detection systems (e.g., analyzing only images) are easily bypassed by deepfakes that manipulate the opposite modality. 

Our approach solves this by explicitly decoupling the detection pipeline into a unified multimodal network. It can pinpoint exactly whether the visual frames are manipulated, whether the audio track is synthetic, or whether both modalities have been altered, providing a comprehensive and robust defense against modern deepfakes.

## Key Features

- **Visual/Image Analysis:** Uses a pre-trained visual encoder to detect spatial inconsistencies and synthetic visual artifacts.
- **Audio Analysis:** Analyzes Log-Mel Spectrograms through a specialized audio encoder to detect voice cloning and synthetic audio.
- **Multimodal Fusion:** Fuses visual and audio embeddings to output a comprehensive final prediction.
- **Modality-Specific Predictions:** Contains three independent prediction heads (Image, Audio, and Fusion) allowing diagnostic isolation of the exact manipulated modality.
- **Grad-CAM Interpretability:** Generates visual heatmaps highlighting the exact regions in the face and frequencies in the audio spectrogram that influenced the model's decision.
- **Flexible Media Support:** Accepts raw video files, standalone images, or standalone audio clips.
- **Streamlit Demonstration:** Features an interactive web interface for running real-time multimodal inference and visualizing Grad-CAM outputs.

## System Architecture

The architecture consists of decoupled unimodal encoders and an independent fusion mechanism:

*   **Video** $\rightarrow$ Visual Encoder $\rightarrow$ **Image Head**
*   **Audio** $\rightarrow$ Audio Encoder $\rightarrow$ **Audio Head**
*   **Visual Features + Audio Features** $\rightarrow$ Fusion Module $\rightarrow$ **Fusion Head**

This multi-task design forces the model to learn independent, modality-specific representations rather than overly relying on a single dominant modality.

## Dataset

This project utilizes the **FakeAVCeleb** dataset, a comprehensive multimodal deepfake dataset containing RealVideo-RealAudio, FakeVideo-RealAudio, RealVideo-FakeAudio, and FakeVideo-FakeAudio samples.

To prevent data leakage and ensure fair evaluation, the dataset was rigorously split using a strict identity-isolation protocol. Identities present in the training set are guaranteed not to appear in the validation or held-out test sets. 

The canonical test split evaluated below consists of exactly **3,270** held-out multimodal samples.

## Results

The following metrics represent the final evaluation on the isolated, held-out test set (3,270 samples).

| Metric | Image Head | Audio Head | Fusion Head |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 99.90% | 99.93% | 99.90% |
| **Precision** | 100.00% | 99.94% | 100.00% |
| **Recall** | 99.90% | 99.94% | 99.90% |
| **F1-Score** | 99.95% | 99.94% | 99.95% |
| **ROC-AUC** | 99.95% | 99.99% | 99.98% |

*(Note: These extraordinary metrics reflect performance strictly on the defined FakeAVCeleb test set distribution and do not necessarily guarantee identical performance on zero-shot in-the-wild manipulations).*

## Research Results

The comprehensive research evidence store is located in the `results/` directory.

### Confusion Matrices
| [Image Head](results/confusion_matrices/image_confusion_matrix.png) | [Audio Head](results/confusion_matrices/audio_confusion_matrix.png) | [Fusion Head](results/confusion_matrices/fusion_confusion_matrix.png) |
|:---:|:---:|:---:|
| ![Image CM](results/confusion_matrices/image_confusion_matrix.png) | ![Audio CM](results/confusion_matrices/audio_confusion_matrix.png) | ![Fusion CM](results/confusion_matrices/fusion_confusion_matrix.png) |

### Performance Curves
*   [**ROC Curves:**](results/roc_curves/) Analyzes threshold-independent classification accuracy across all three heads.
*   [**Precision-Recall Curves:**](results/precision_recall_curves/) Validates model robustness against the natural class imbalances present in FakeAVCeleb.
*   [**Training History:**](results/performance_curves/training_history.json) Tracks the multi-task loss and convergence over 10 epochs.

### Modality Analysis & Error Analysis
*   [**Four-Category Modality Diagnostic:**](results/modality_analysis/modality_category_results.csv) Proves the model successfully identifies partial manipulations (e.g., detecting fake audio overlaid on authentic video).
*   [**Error Analysis:**](results/error_analysis/) Granular reports of the exact False Positives and False Negatives produced by the Fusion head.

### Grad-CAM Interpretability
Visual heatmaps demonstrating the model's focus regions are cataloged by category:
*   [FakeVideo + FakeAudio](results/gradcam/FakeVideo-FakeAudio/)
*   [FakeVideo + RealAudio](results/gradcam/FakeVideo-RealAudio/)
*   [RealVideo + FakeAudio](results/gradcam/RealVideo-FakeAudio/)
*   [RealVideo + RealAudio](results/gradcam/RealVideo-RealAudio/)

## Demo

The interactive Streamlit application allows you to upload media and view the independent predictions and Grad-CAM visualizations.

To run the application locally:

```bash
streamlit run app.py
```

---
*For a complete historical breakdown, technical details, and implementation notes, please refer to [PROJECT_FULL_DOC.md](PROJECT_FULL_DOC.md).*
