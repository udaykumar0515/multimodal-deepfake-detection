# PROJECT_FULL_DOC.md — Deepfake Detection Using Multimodal Learning
## Complete Technical Documentation

> **Purpose of This Document**
> This is the authoritative long-term memory for the "Deepfake Detection Using Multimodal Learning" student research/engineering project. It allows the original author — or any reader — to fully reconstruct the project's purpose, evolution, implementation, experiments, results, and limitations without reading every source file or the raw project log.
>
> The companion `README.md` provides a concise public-facing summary. This document provides the complete technical and historical record.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Evolution — Development History](#3-project-evolution--development-history)
4. [Final System Architecture](#4-final-system-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Dataset — FakeAVCeleb](#6-dataset--fakeavceleb)
7. [Data Splitting and Leakage Prevention](#7-data-splitting-and-leakage-prevention)
8. [Visual Preprocessing](#8-visual-preprocessing)
9. [Audio Preprocessing](#9-audio-preprocessing)
10. [Dataset/DataLoader Implementation](#10-datasetdataloader-implementation)
11. [Loss Function and Training Strategy](#11-loss-function-and-training-strategy)
12. [Training and Checkpoint Selection](#12-training-and-checkpoint-selection)
13. [Evaluation Methodology](#13-evaluation-methodology)
14. [Final Results](#14-final-results)
15. [Error Analysis](#15-error-analysis)
16. [Grad-CAM / Interpretability](#16-grad-cam--interpretability)
17. [Streamlit Application](#17-streamlit-application)
18. [Testing/Demo Data](#18-testingdemo-data)
19. [Historical V1 Architecture (Reference)](#19-historical-v1-architecture-reference)
20. [Limitations](#20-limitations)
21. [Future Work](#21-future-work)
22. [Setup and Reproduction](#22-setup-and-reproduction)
23. [Interview / Viva Memory](#23-interview--viva-memory)
24. [Final Project State](#24-final-project-state)

---

## 1. Project Overview

### 1.1 Purpose and Summary

This project implements a **Multi-Task Multimodal Deepfake Detection** system capable of independently detecting manipulation in visual (image/video) and audio streams. The system produces three separate predictions for any given video input: one from a dedicated visual prediction head, one from a dedicated audio prediction head, and one from a fusion head that combines both modalities. These predictions can disagree — a property that is the project's primary scientific contribution.

The final system is packaged as:
- A fully trained PyTorch model checkpoint (`checkpoints/best_model.pt`)
- A comprehensive research results store (`results/`)
- An interactive Streamlit web application (`app.py`)

### 1.2 Key Objectives

1. Implement a reproducible, identity-isolated dataset pipeline for the FakeAVCeleb dataset.
2. Design and train a multi-task architecture with independent visual, audio, and fusion prediction heads.
3. Demonstrate that the visual and audio branches learn genuinely decoupled representations — i.e., the audio head can detect fake audio even when paired with authentic video, and vice versa.
4. Generate interpretability evidence (Grad-CAM) for each modality branch.
5. Build an interactive demonstration application supporting single-image, audio-only, and full video+audio inference.

### 1.3 Core Contributions

| Contribution | Type | Status |
|---|---|---|
| Identity-isolated canonical dataset manifests with full audit trail | Engineering | Complete |
| Offline face-detection preprocessing with fallback hierarchy | Engineering | Complete |
| Offline audio extraction and log-mel spectrogram generation | Engineering | Complete |
| Multi-task multi-head model (Visual + Audio + Fusion heads) | Architecture | Complete |
| Multi-task Binary Focal Loss with equal head weighting | Training | Complete |
| Four-category modality decoupling diagnostic | Research Evaluation | Complete |
| Modality-specific Grad-CAM interpretability | Research Evidence | Complete |
| Interactive Streamlit demonstration application | Application | Complete |

---

## 2. Problem Statement

### 2.1 The Deepfake Threat

Deepfake technology refers to the use of generative AI — most commonly GAN-based or diffusion-based approaches — to synthesize convincing but fabricated audiovisual content. Prominent examples include face-swapped video, voice-cloned audio, and entirely synthetic talking-head sequences. The proliferation of such content poses genuine risks in contexts ranging from political misinformation to personal harassment and identity fraud.

### 2.2 Why Unimodal Detection Fails

Detection systems that analyze only one modality are structurally fragile against adversaries who manipulate the other. Specifically:

- A **visual-only** detector will fail entirely on deepfakes where only the audio has been synthesized (e.g., a real video with cloned voice-over).
- An **audio-only** detector will fail on purely visual deepfakes with authentic audio.
- A detector that simply **concatenates** features before any prediction cannot produce independent verdicts — it cannot tell you *which* modality was manipulated.

This is not merely a theoretical concern. The V1 single-head architecture implemented in this project confirmed the problem empirically: the single fusion model suffered from **visual dominance**, meaning it effectively ignored audio cues and was structurally unable to differentiate between `RealVideo+FakeAudio` and `RealVideo+RealAudio` categories.

### 2.3 Why Multimodal Learning — and Why Independent Heads

The project's answer is a **multi-task architecture** with three decoupled prediction heads:

1. An **Image Head** that predicts deepfake probability from visual features alone.
2. An **Audio Head** that predicts deepfake probability from audio features alone.
3. A **Fusion Head** that predicts an overall probability from the joint representation.

By training all three heads simultaneously with independent losses, the network is forced to build genuinely separate representations for each modality. The scientific value is demonstrated by the **four-category modality diagnostic**: on samples containing real video but fake audio, the image head correctly predicts REAL while the audio head independently predicts FAKE.

### 2.4 Project Scope and Constraints

- **Dataset:** FakeAVCeleb (locally stored; not committed to the repository).
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
- **Evaluation:** Strictly on the frozen held-out FakeAVCeleb test split. Results are **not** claimed to generalize to all real-world deepfakes.
- **Application:** Demonstration-grade Streamlit application, not a production deployment.

---

## 3. Project Evolution — Development History

This section reconstructs the project's documented development history from `PROJECT_LOG.md`. Each sub-section corresponds to a documented phase or stage.

- Sections marked **[HISTORICAL]** describe work that has been superseded in the final system.
- Sections marked **[FINAL]** describe work that is directly part of the active project.

---

### 3.1 Governance and Baseline (2026-09-06) [FINAL — Infrastructure]

**Objective:** Establish project governance and confirm the starting state of the repository before any implementation.

**Work performed:**
- Confirmed GitHub remote and that the `main` branch was at commit `f8eed83`.
- Confirmed `.gitignore` excluded the raw dataset directory, notebook checkpoints, and Python caches.
- Created `PROJECT_LOG.md` as the permanent implementation record.
- Reviewed and adopted the implementation blueprint as the governing policy.

**Key decision:** Do not begin dataset splitting with assumed counts — reconcile the metadata first. The audit identified a discrepancy (1,500 vs 500 vs 1,000 real sample counts in different documents). This was flagged and held for resolution in Phase 1.

**Resulting state:** Governance confirmed. No code changed.

---

### 3.2 Phase 1: Dataset Preparation — Identity Split (2026-09-06) [FINAL]

**Objective:** Implement reproducible identity-based train/validation/test CSV manifests.

**Why identity-based splitting:** Random per-video splitting would allow the same subject identity to appear in both training and test sets. Because deepfake models encode identity-specific artefacts, this constitutes data leakage and inflates generalization metrics.

**Work performed:**
- Read `Dataset_FakeAVCeleb/meta_data.csv`: 21,566 metadata rows, 500 unique source identities.
- Assigned identities using fixed seed `42`: 350 train / 75 validation / 75 test.
- Labels derived: `video_label` from `RealVideo-* → Real`, `FakeVideo-* → Fake`; `audio_label` retained from audio component of `type`.
- Created `scripts/create_identity_split.py` → generated `data/splits/train.csv`, `val.csv`, `test.csv`.

**Verification:** SHA-256 hashes stable across regenerations. Zero cross-split identity/path overlap. All 21,566 rows assigned exactly once.

**Final partition sizes:**
| Split | Identities | Rows | Real | Fake |
|---|---|---|---|---|
| Train | 350 | 15,099 | 700 | 14,399 |
| Validation | 75 | 3,194 | 150 | 3,044 |
| Test | 75 | 3,273 | 150 | 3,123 |

**Problem discovered:** 22 physical video paths had two metadata rows each (differing only in `method` annotation: `faceswap-wav2lip` vs `wav2lip`). Zero label conflicts.

**Decision:** Do not silently deduplicate. Retain all rows. Address in subsequent audit.

---

### 3.3 Phase 1 (continued): Duplicate Record Audit (2026-09-06) [FINAL — Informational]

**Objective:** Classify the 22 duplicated video paths before any preprocessing.

**Findings:** Every duplicate group occurred within exactly one split. Zero label conflicts. The duplicates represent the same physical video annotated under two manipulation method labels.

**Decision:** Keep original CSVs unmodified. Produce canonical one-row-per-video manifests as a separate layer for all downstream preprocessing.

---

### 3.4 Phase 1: Final Dataset Freeze and Canonical Manifests (2026-09-06) [FINAL]

**Objective:** Create canonical manifests that preprocessing would consume while preserving full provenance.

**Work performed:**
- `scripts/create_canonical_manifests.py`: one row per physical video, both method annotations stored in `method_annotations`, all original rows preserved in JSON `metadata_provenance`.
- `scripts/validate_dataset.py`: validated all 21,544 canonical videos for existence and OpenCV frame readability — **0 failures**.
- Generated `data/splits/canonical/train.csv`, `val.csv`, `test.csv`.
- Git commit: `c863a8c`.

**Final canonical statistics:**
| Split | Videos | Real | Fake |
|---|---|---|---|
| Train | 15,083 | 700 | 14,383 |
| Validation | 3,191 | 150 | 3,041 |
| Test | 3,270 | 150 | 3,120 |

**Resulting state:** Dataset frozen. All future preprocessing must use canonical manifests only.

---

### 3.5 Phases 1.5–1.8: Video Preprocessing Experiments [HISTORICAL — Informed Final Pipeline]

These phases represent iterative experimentation to validate a viable face-detection strategy before committing to full-scale preprocessing. The findings directly shaped the final offline pipeline.

**Phase 1.5 — Initial quality validation (50-video CPU sample):**
- InsightFace RetinaFace (`det_10g`) on CPU.
- Direct detection success rate: **41.5%** | Center-crop fallback: **58.5%**
- Conclusion: 58.5% blind center-crop rate unacceptable. Full-scale processing halted.

**Phase 1.6 — GPU environment diagnosis:**
- Root cause: Python environment had CPU-only `torch` and `onnxruntime` despite hardware support.
- Fix: Reinstalled CUDA-capable packages.

**Phase 1.6.1 — GPU acceleration validation:**
- Speed improved ~3.2× (~4.57 → ~1.42 sec/video).
- Face detection rate remained at **41.5%** — confirming failures are data/model characteristics, not compute precision.

**Phase 1.7 — Bounding-box forward-filling experiment:**
- Stateful forward-fill: when a frame misses detection, reuse the most recent successful bounding box.
- Center-crop fallback reduced: **58.5% → 33.1%**. Forward-fill: 25.4%.
- Fully-covered videos (zero fallbacks) rose to 20/50.

**Phase 1.8 — Forward-fill quality check:**
- Mean fill streak: ~3 frames. Max streak: 12 frames. Mean geometric drift: **1.63%** (negligible given 20% margin).
- Remaining 33.1% fallbacks occur at video starts where no prior bounding box exists.
- Backward-filling considered and explicitly **excluded** from the frozen methodology.

**Resulting state:** Forward-fill strategy validated and adopted. Backward-filling permanently excluded.

---

### 3.6 Phase 1.10: Repository Cleanup #1 [HISTORICAL — Organizational]

Validation scripts, GPU diagnostics, and experimental reports consolidated into a `testing/` hierarchy (later renamed `archive/` in Phase 9.5). Active `scripts/` and `data/` directories retained only production-ready files.

---

### 3.7 Phase 2: Full-Scale Video Preprocessing (Completed) [FINAL]

**Objective:** Run the finalized face-detection pipeline across all 21,544 videos.

**Configuration:** `UniformTemporalSampler` (16 frames/video), RetinaFace (`det_10g`), 20% margin, 224×224. Fallback hierarchy: Direct Detection → Forward-Fill → Center-Crop.

**Results:**
- Videos processed: **21,544 / 21,544 (0 failures)**
- Frames extracted: **344,704** (Direct: 141,166 | Forward-fill: 83,946 | Center-crop: 119,592)
- Output: `.npy` arrays `[16, 224, 224, 3]` uint8 | Size: ~48.33 GB | Runtime: ~9h 12m

**Key notes:** GPU confirmed active. ONNX Runtime `LoadLibrary failed with error 126` warning is a harmless initialization artefact. Raw dataset files verified completely untouched.

---

### 3.8 Phase 3: Audio Preprocessing (Completed) [FINAL]

**Objective:** Extract audio and generate log-mel spectrograms as offline `.npy` tensors.

**Configuration:**
- Extraction: `imageio-ffmpeg` FFmpeg binary → 16 kHz, mono, 16-bit PCM WAV.
- Spectrogram: 128-mel, `n_fft=1024`, `hop_length=512`, `f_min=20 Hz`, `f_max=8000 Hz`, dB scale, bilinear resize to 224×224, 3-channel repeat → `float32 [3, 224, 224]`.

**Results:** 21,544 / 21,544 success (0 failures). Output: ~15.51 GB. Runtime: ~46m 14s.

**Dependency note:** `soundfile` must be installed for torchaudio on Windows (`pip install soundfile`). `imageio-ffmpeg` provides an isolated FFmpeg binary.

---

### 3.9 Phase 4: V1 PyTorch Dataset and DataLoader [HISTORICAL — Core Patterns Retained]

The V1 dataset implementation established design patterns carried into the final V2 system:

- **Lazy `.npy` loading** in `__getitem__` (not pre-loaded into RAM).
- **Temporally consistent augmentation** via `albumentations` `additional_targets` — all 16 frames in a sample receive identical spatial transforms.
- **`WeightedRandomSampler`** for training: ~4.6% Real samples in training → inverse-frequency weights → approximately 50/50 balanced batches.
- **Natural distribution** for validation and test (no resampling).
- **Labels:** `Real → 0.0`, `Fake → 1.0` (float32).

---

### 3.10 Phase 5: V1 Architecture — Single-Head Model [HISTORICAL]

**Architecture (V1):**
- `VideoEncoder`: EfficientNet-B0, mean-pools 16 frames → `(B, 1280)`
- `AudioEncoder`: EfficientNet-B0 for spectrograms → `(B, 1280)`
- `FusionModel`: concat `(B, 2560)` → `Linear→ReLU→Dropout→Linear` → single scalar logit
- Parameters: 9,326,841 trainable

**Status:** Superseded by V2. All V1 artefacts in `archive/v1_single_head/`.

---

### 3.11 Phases 6–6.5: V1 Training Infrastructure and Hardware Timing [HISTORICAL]

**Training infrastructure (V1):**
- `BinaryFocalLoss` (`gamma=2.0`) — **retained unchanged in final V2 system**
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- Scheduler: CosineAnnealingLR
- AMP + gradient clipping (`max_norm=1.0`)
- Checkpointing: `best_model.pt` + `latest.pt`

**Hardware timing (RTX 4050, batch_size=4):**
- Throughput: 20.94 videos/sec | Peak VRAM: 3.03 GB / 6 GB
- Estimated 10-epoch time: ~2h 7m | VRAM headroom: ~2.73 GB

**Windows note:** `num_workers=4` caused post-training DataLoader crash in timing test; resolved with `num_workers=0` in timing script only. Actual `train.py` used `num_workers=4` safely.

---

### 3.12 Phases 7–8: V1 Training and Evaluation Preparation [HISTORICAL]

V1 training was executed and `scripts/evaluate_test.py` was created. Specific V1 test metrics are not documented in the available materials, as Phase 10.5 revealed V1's fundamental architectural limitation, making those results scientifically incomplete. All V1 artefacts preserved in `archive/v1_single_head/`.

---

### 3.13 Phases 9A–9B: V1 Grad-CAM [HISTORICAL]

**Phase 9A (sanity check):** Verified that hooks could be registered on `backbone.features[-1]` of both EfficientNet encoders. Activation shapes: `[16, 1280, 7, 7]` (video), `[1, 1280, 7, 7]` (audio).

**Phase 9B (implementation):** `GradCAM` class implemented: global average pooling over gradients, weighted activation sum, ReLU, `[0,1]` normalization, spatial upsampling, `COLORMAP_JET` overlay. Visualizations for 5 test samples (correct fake, 3 false negatives, correct real). Outputs archived.

**Constraint established here and maintained throughout:** Grad-CAM provides attribution — it does not constitutively prove the presence of a specific physical deepfake artefact.

---

### 3.14 Phase 9.5: Repository Cleanup #2 [HISTORICAL — Organizational]

`testing/` renamed → `archive/`. One-time training analysis and Grad-CAM scripts moved to `archive/`. V1 Grad-CAM outputs moved from `results/gradcam/` to `archive/gradcam/outputs/`. `scripts/` reduced to the four core active entry points.

---

### 3.15 Phase 10: V1 Streamlit Application [HISTORICAL]

Initial `app.py` with Detection and Results & Evaluation pages. Loaded V1 `best_model.pt`, performed video inference with inline Grad-CAM. Superseded by the V2 application (Phase 17).

---

### 3.16 Phase 10.5: The Critical Architectural Decision — V1 → V2 [HISTORICAL → FINAL]

**Conclusion:** Extending V1 to produce independent per-modality predictions was scientifically and technically impossible. V1's single fusion logit was inherently incapable of isolating visual from audio evidence. Visual dominance was confirmed: the `RealVideo+FakeAudio` category was structurally invisible to V1.

**Decision:** Archive all V1 system files and rebuild from scratch as a multi-head multi-task architecture. Preprocessing infrastructure (`data/`, `preprocessing/`) preserved intact — no re-preprocessing required.

This transition permanently defines the V1 (historical) / V2 (final/active) boundary.

---

### 3.17 Phases 11–14: V2 Multi-Head Architecture — Design to Training-Ready [FINAL]

**Phase 11 — Architecture:**
- `VisualEncoder`: EfficientNet-B0, returns mean-pooled `(B, 1280)` or unpooled `(B×T, 1280)` frame features.
- `AudioEncoder`: EfficientNet-B0, spectrogram input → `(B, 1280)`.
- `MultiHeadDeepfakeModel`: assembles both encoders with three independent heads.

**Phase 12 — Dataset:**
- `MultimodalDeepfakeDataset` returns `(video, audio, video_label, audio_label, overall_label)`.
- Image-head training: `video_label` dynamically expanded via `repeat_interleave(16, dim=0)` to align with unpooled frame predictions — no separate image dataset required.

**Phase 13 — Loss and trainer:**
- `MultiTaskFocalLoss`: equal unweighted sum of per-head focal losses.
- `Trainer`: tracks all three head metrics per epoch; uses `ReduceLROnPlateau` (changed from V1's CosineAnnealingLR).

**Phase 14 — Training verification:**
- `scripts/train.py`: unified entry point. Configuration: `batch_size=4`, `lr=1e-4`, `wd=1e-4`, `gamma=2.0`, `grad_clip=1.0`, AMP.

---

### 3.18 Phase 15: V2 Final Training and Test Evaluation [FINAL]

**Training:** 10 epochs on RTX 4050. Best checkpoint: **Epoch 6** (`total_val_loss=0.0189`, Fusion Acc 99.81%).

**Key finding — four-category modality diagnostic:**

| Category | N | Image Acc | Audio Acc | Fusion Acc | Image Fake Prob | Audio Fake Prob |
|---|---|---|---|---|---|---|
| RealVideo + RealAudio | 75 | 100% | 100% | 100% | — | 1.06% |
| FakeVideo + RealAudio | 1,471 | 99.80% | 99.93% | 99.80% | — | 0.64% |
| RealVideo + FakeAudio | 75 | 100% | 100% | 100% | 5.44% | 98.92% |
| FakeVideo + FakeAudio | 1,649 | 100% | 99.94% | 100% | — | — |

On `RealVideo + FakeAudio`: Image Head correctly predicted REAL (5.44% fake prob) while Audio Head independently predicted FAKE (98.92% fake prob). This empirically demonstrates successful modality decoupling.

---

### 3.19 Phase 16: V2 Grad-CAM Generation [FINAL]

`scripts/generate_gradcam.py`: Visual Grad-CAM backpropagated from Image Head; Audio Grad-CAM from Audio Head. One representative sample per four-category combination from the held-out test set. Overlays on frames `[0, 7, 15]` and audio spectrogram. Outputs organized by category in `results/gradcam/`.

---

### 3.20 Phases 17–22: Application, Cleanup, and Documentation [FINAL]

- **Phase 17:** Final `app.py` — Video/Image/Audio inference, inline Grad-CAM, Results page.
- **Phase 18:** V1 assets archived. `testing_data/` structure created.
- **Phase 19:** `testing_data/` populated with 4 samples per category from the held-out test set.
- **Phase 20:** All `v2_` prefixes removed from active files. `results/` reorganized into research subdirectories. PR curves and error analysis CSVs generated.
- **Phase 21:** `scripts/` reduced to files with demonstrable active roles; `generate_gradcam.py` retained for research reproducibility.
- **Phase 22:** `README.md` authored. `PROJECT_FULL_DOC.md` initiated.
- **archive/ added to `.gitignore`:** Archive directory is local-only.

---

---

## 4. Final System Architecture

### 4.1 Architecture Overview

The final model is `MultiHeadDeepfakeModel`, a multi-task neural network with two independent EfficientNet-B0 backbone encoders (one visual, one audio) and three independent prediction heads. The model's defining property is that it can produce separate, potentially disagreeing predictions for each modality, which is the core scientific contribution of the project.

```
INPUT MODALITIES
────────────────────────────────────────────────────────────────
Video: (B, T=16, 3, 224, 224)         Audio: (B, 3, 224, 224)
        │                                       │
        ▼                                       ▼
┌──────────────────┐                 ┌──────────────────────┐
│  VisualEncoder   │                 │    AudioEncoder      │
│  EfficientNet-B0 │                 │    EfficientNet-B0   │
│                  │                 │                      │
│  Fold T into B   │                 │  features + avgpool  │
│  (B×T, 3,224,224)│                 │       ↓              │
│  features+avgpool│                 │  (B, 1280)           │
│  (B×T, 1280)     │                 └────────┬─────────────┘
│  mean-pool or    │                          │
│  return unpooled │                          │
│  (B, 1280)       │                          │
└──────┬───────────┘                          │
       │ seq_rep (B, 1280)                    │ audio_feat (B, 1280)
       │ frame_feats (B×T, 1280) [training]   │
       │                                      │
       ├──────────────────────────────────────┤
       │           concat: (B, 2560)          │
       │                                      │
       ▼                                      │
┌─────────────────────────────┐              │
│       fusion_head           │              │
│  Linear(2560→512)           │              │
│  ReLU → Dropout(0.3)        │              │
│  Linear(512→1)              │              │
│  → fusion logit             │              │
└─────────────────────────────┘              │
                                             ▼
                              ┌──────────────────────────┐
                              │       audio_head         │
                              │   Dropout(0.3)           │
                              │   Linear(1280→1)         │
                              │   → audio logit          │
                              └──────────────────────────┘

       frame_feats (B×T, 1280) [training only]
       │
       ▼
┌──────────────────────────┐
│       image_head         │
│   Dropout(0.3)           │
│   Linear(1280→1)         │
│   → (B×T, 1) frame logits│
└──────────────────────────┘
```

All heads output **raw logits** (no sigmoid inside the model). Sigmoid is applied externally by the loss function during training (`BCEWithLogitsLoss`-based Focal Loss) and explicitly during inference.

---

### 4.2 VisualEncoder

**File:** `models/visual_encoder.py`

**Purpose:** Encodes image or video input into a 1280-dimensional feature vector using a pretrained EfficientNet-B0 backbone.

**Constructor:** `VisualEncoder(pretrained=True)` loads `EfficientNet_B0_Weights.IMAGENET1K_V1`. Retains `backbone.features` (the convolutional feature extractor) and `backbone.avgpool` (AdaptiveAvgPool2d → `[1, 1]`). Exposes `self.grad_cam_layer = self.features[-1]` (the final MBConv block) for Grad-CAM hook registration.

**Forward dispatch:** Automatically detects input dimensionality:
- **4D input** `(B, 3, 224, 224)` → `forward_image` → `(B, 1280)`
- **5D input** `(B, T, 3, 224, 224)` → `forward_video` → `(B, 1280)` [or `(B, 1280) + (B×T, 1280)` if `return_unpooled=True`]

**Video processing detail:**
1. Reshape: `(B, T, 3, 224, 224)` → `(B×T, 3, 224, 224)` (fold time into batch dimension)
2. Forward through `features + avgpool + flatten`: → `(B×T, 1280)` per-frame features
3. Unfold: `(B×T, 1280)` → `(B, T, 1280)` → mean over T → `(B, 1280)` sequence representation
4. If `return_unpooled=True`: return both `(B, 1280)` and raw `(B×T, 1280)`

**Why `return_unpooled` matters:** During multi-task training, the Image Head receives the `(B×T, 1280)` unpooled frame features, so it learns to classify each frame independently rather than only the mean-pooled video summary.

---

### 4.3 AudioEncoder

**File:** `models/audio_encoder.py`

**Purpose:** Encodes a 3-channel 224×224 log-mel spectrogram into a 1280-dimensional feature vector.

**Constructor:** `AudioEncoder(pretrained=True)` loads `EfficientNet_B0_Weights.IMAGENET1K_V1`. Retains `backbone.features` and `backbone.avgpool`. Exposes `self.grad_cam_layer = self.features[-1]` for Grad-CAM.

**Forward:** `(B, 3, 224, 224)` → `features` → `avgpool` → `flatten` → `(B, 1280)`

**Design rationale for treating spectrogram as image:** The log-mel spectrogram is a 2D time-frequency representation. Treating it as a 3-channel image (via channel duplication) allows the same pretrained EfficientNet-B0 to extract frequency and temporal patterns without requiring a separate audio-specific backbone. This is a pragmatic engineering choice motivated by the availability of strong pretrained image features.

---

### 4.4 MultiHeadDeepfakeModel — Forward Pass Logic

**File:** `models/multihead_model.py`

**Constructor:** `MultiHeadDeepfakeModel(pretrained=True, dropout=0.3)`

The `forward` method enforces strict input routing via explicit combinatorial checks. Passing an invalid combination raises `ValueError` rather than silently producing incorrect results.

**Routing logic:**

| Inputs provided | Route | Output keys |
|---|---|---|
| `image` only (no video, no audio) | `VisualEncoder.forward_image` → `image_head` | `{'image': logit}` |
| `audio` only (no video, no image) | `AudioEncoder` → `audio_head` | `{'audio': logit}` |
| `video` + `audio`, `return_all=False` | `VisualEncoder` (pooled) + `AudioEncoder` → `fusion_head` | `{'fusion': logit}` |
| `video` + `audio`, `return_all=True` | Both encoders; unpooled frame features → `image_head`; `audio_feat` → `audio_head`; concatenated → `fusion_head` | `{'fusion', 'audio', 'image'}` |

`return_all=True` is used during training and is computationally complete. `return_all=False` is used for fusion-only inference.

---

### 4.5 Prediction Heads

| Head | Input Shape | Architecture | Output Shape |
|---|---|---|---|
| `image_head` | `(B×T, 1280)` | `Dropout(0.3) → Linear(1280, 1)` | `(B×T, 1)` |
| `audio_head` | `(B, 1280)` | `Dropout(0.3) → Linear(1280, 1)` | `(B, 1)` |
| `fusion_head` | `(B, 2560)` | `Linear(2560, 512) → ReLU → Dropout(0.3) → Linear(512, 1)` | `(B, 1)` |

The fusion head is deeper than the per-modality heads because it must learn to integrate two heterogeneous feature spaces (visual and audio) rather than specializing in a single modality.

**Sigmoid and thresholding:** Applied externally. During inference: `prob = torch.sigmoid(logit)`. Classification threshold: `0.5` (Fake if `prob ≥ 0.5`).

---

### 4.6 Model Parameter Count

From the project log Phase 5 sanity test (architecture unchanged in V2):
- **Total trainable parameters:** 9,326,841
- Both EfficientNet-B0 backbones are loaded with ImageNet pretrained weights. All parameters are trainable during fine-tuning (no frozen layers).

---

### 4.7 Grad-CAM Target Layer

Both encoders expose `self.grad_cam_layer` pointing to `self.features[-1]`, which is the final `MBConv` block of EfficientNet-B0. This produces intermediate activation maps of shape `[B, 1280, 7, 7]` (for 224×224 inputs), which are the spatial feature maps used in the Grad-CAM computation.

---

## 5. Repository Structure

### 5.1 Active Project Tree

```
Deepfake_Detection/                     ← Repository root
│
├── app.py                              ← Streamlit application entry point
│
├── models/                             ← Final model architecture
│   ├── visual_encoder.py               ← VisualEncoder (EfficientNet-B0)
│   ├── audio_encoder.py                ← AudioEncoder (EfficientNet-B0)
│   ├── multihead_model.py              ← MultiHeadDeepfakeModel
│   └── __init__.py
│
├── dataset/                            ← PyTorch dataset and dataloader
│   ├── multimodal_dataset.py           ← MultimodalDeepfakeDataset
│   ├── dataloader_factory.py           ← build_dataloaders() factory
│   └── __init__.py
│
├── preprocessing/                      ← Preprocessing classes (used offline + in app)
│   ├── video_preprocessing.py          ← UniformTemporalSampler, RetinaFaceCropper,
│   │                                      VideoPreprocessor, VisualTransform
│   ├── audio_preprocessing.py          ← AudioExtractor, SpectrogramGenerator
│   └── __init__.py
│
├── training/                           ← Training loop and losses
│   ├── trainer.py                      ← Trainer class (multi-task)
│   ├── losses.py                       ← BinaryFocalLoss, MultiTaskFocalLoss
│   └── __init__.py
│
├── scripts/                            ← Active executable entry points
│   ├── train.py                        ← Full training run
│   ├── evaluate_test.py                ← Held-out test evaluation
│   ├── preprocess_dataset_offline.py   ← Offline video frame preprocessing
│   ├── preprocess_audio_offline.py     ← Offline audio spectrogram preprocessing
│   └── generate_gradcam.py             ← Grad-CAM visualization generation
│
├── checkpoints/                        ← Model checkpoints (git-ignored)
│   ├── best_model.pt                   ← Best validation loss checkpoint (Epoch 6)
│   └── latest_model.pt                 ← Last completed epoch checkpoint
│
├── results/                            ← Research evidence store
│   ├── metrics/
│   │   └── test_metrics.json           ← Final per-head test metrics
│   ├── predictions/
│   │   └── test_predictions.csv        ← Per-sample predictions + probabilities
│   ├── confusion_matrices/             ← image, audio, fusion confusion matrix PNGs
│   ├── roc_curves/                     ← image, audio, fusion ROC curve PNGs
│   ├── precision_recall_curves/        ← Per-head PR curve PNGs
│   ├── performance_curves/
│   │   └── training_history.json       ← Per-epoch loss and metric history
│   ├── modality_analysis/
│   │   └── modality_category_results.csv
│   ├── error_analysis/                 ← False positive / false negative CSVs
│   ├── gradcam/                        ← Grad-CAM overlays by category
│   ├── dataset_summary.json
│   └── README.md                       ← Research evidence index
│
├── data/                               ← Processed dataset (git-ignored)
│   ├── dataset_split/
│   │   ├── train.csv                   ← Canonical training manifest
│   │   ├── val.csv                     ← Canonical validation manifest
│   │   └── test.csv                    ← Canonical test manifest
│   ├── processed_frames/               ← Offline .npy video frame arrays (~48 GB)
│   ├── processed_audio/                ← Offline .npy spectrogram arrays (~15 GB)
│   └── README.md
│
├── testing_data/                       ← Demo samples for Streamlit testing
│   ├── images/real/ and images/fake/
│   ├── audio/real/ and audio/fake/
│   ├── videos/<four categories>/
│   ├── TESTING_DATA_MANIFEST.csv
│   └── README.md
│
├── docs/                               ← Project documentation files
├── archive/                            ← Historical/superseded artefacts (git-ignored)
│
├── PROJECT_LOG.md                      ← Permanent development history
├── PROJECT_FULL_DOC.md                 ← This document
├── README.md                           ← Public-facing repository summary
├── .gitignore
└── .env                                ← PYTHONPATH=. (sets project root on path)
```

### 5.2 Key File Relationships

- `app.py` imports from `models/`, `preprocessing/`, and reads from `results/` and `checkpoints/`.
- `scripts/train.py` orchestrates `dataset/`, `models/`, and `training/`.
- `scripts/evaluate_test.py` loads `checkpoints/best_model.pt`, uses `dataset/`, writes to `results/`.
- `scripts/generate_gradcam.py` loads `checkpoints/best_model.pt`, uses `preprocessing/` and `data/dataset_split/`, writes to `results/gradcam/`.
- `preprocessing/` classes are shared between the offline scripts and the live Streamlit inference pipeline. This ensures identical preprocessing is applied during training-time preprocessing and at inference time.

### 5.3 git-ignored Paths

The following directories and their contents are not committed to the remote repository:
- `.venv_gpu/` — Python virtual environment
- `data/processed_frames/` — ~48 GB offline video frames
- `data/processed_audio/` — ~15 GB offline spectrograms
- `checkpoints/` — trained model weights
- `archive/` — historical artefacts
- `.vscode/` — editor configuration
- `Dataset_FakeAVCeleb/` — raw dataset

---

## 6. Dataset — FakeAVCeleb

### 6.1 Overview and Source

**Name:** FakeAVCeleb  
**Type:** Multimodal (audio-visual) deepfake detection benchmark dataset  
**Storage:** Local filesystem only. Not committed to the repository. Referenced via the path `Dataset_FakeAVCeleb/` relative to the project root.

FakeAVCeleb is a comprehensive audiovisual deepfake dataset containing videos with four distinct manipulation categories, making it specifically suitable for studying multimodal deepfake scenarios:

| Category | Description |
|---|---|
| `RealVideo-RealAudio` | Authentic video and authentic audio (genuine footage) |
| `FakeVideo-RealAudio` | Face-swapped/synthesized video with the original authentic audio |
| `RealVideo-FakeAudio` | Authentic video with synthesized/cloned audio |
| `FakeVideo-FakeAudio` | Both visual and audio streams manipulated |

This four-category structure is what makes FakeAVCeleb uniquely suited to the project's scientific goal: demonstrating that independent modality heads learn separate, decoupled representations.

### 6.2 Composition and Labels

**Raw metadata file:** `Dataset_FakeAVCeleb/meta_data.csv`

- **Total metadata rows:** 21,566
- **Unique physical video paths:** 21,544 (22 paths appear in two rows each — see Section 6.4)
- **Unique source identities:** 500 (the `source` field)
- **`type` field:** Encodes the four-category label (`RealVideo-RealAudio`, `FakeVideo-RealAudio`, `RealVideo-FakeAudio`, `FakeVideo-FakeAudio`)
- **`method` field:** Records the manipulation technique. Known values include `wav2lip` and `faceswap-wav2lip`.

**Label derivation:**

| `type` prefix | `video_label` | `audio_label` | Primary `label` |
|---|---|---|---|
| `RealVideo-*` | Real | depends on suffix | Real |
| `FakeVideo-*` | Fake | depends on suffix | Fake |
| `*-RealAudio` | depends on prefix | Real | — |
| `*-FakeAudio` | depends on prefix | Fake | — |

The primary `label` (used for the Fusion Head and WeightedRandomSampler) mirrors `video_label`. The multi-task training additionally uses `audio_label` independently to supervise the Audio Head.

### 6.3 Canonical Dataset Statistics

After identity-based splitting and deduplication (one row per physical video):

| Split | Identities | Videos | Real Videos | Fake Videos | Real % |
|---|---|---|---|---|---|
| **Train** | 350 | 15,083 | 700 | 14,383 | 4.64% |
| **Validation** | 75 | 3,191 | 150 | 3,041 | 4.70% |
| **Test** | 75 | 3,270 | 150 | 3,120 | 4.59% |
| **Total** | 500 | 21,544 | 1,000 | 20,544 | 4.64% |

**Class imbalance note:** The dataset is severely imbalanced (~95% Fake, ~5% Real). This is addressed during training by `WeightedRandomSampler` (producing balanced batches) and `BinaryFocalLoss` (down-weighting easy examples). Validation and test loaders use the natural unbalanced distribution to report realistic performance estimates.

**Test split category breakdown** (from `results/dataset_summary.json`):

| Category | Count |
|---|---|
| RealVideo-RealAudio | 75 |
| FakeVideo-RealAudio | 1,471 |
| RealVideo-FakeAudio | 75 |
| FakeVideo-FakeAudio | 1,649 |
| **Total** | **3,270** |

### 6.4 Duplicate Record Policy

The raw metadata contains 44 rows referencing 22 physical video paths twice each. Each duplicate pair differs only in the `method` field (`faceswap-wav2lip` vs `wav2lip`), with no label conflicts.

**Resolution:**
- The original split CSVs (`data/splits/`) retain all 21,566 rows for complete provenance.
- The canonical manifests (`data/dataset_split/`) contain exactly one row per physical video. The two method annotations are stored in a `method_annotations` column; both original rows are preserved in a JSON `metadata_provenance` field.
- All preprocessing was performed using the canonical manifests only, ensuring each physical video was processed exactly once.

### 6.5 Manifest Layers

| File | Purpose | Row count |
|---|---|---|
| `data/splits/train.csv` | Original split — provenance layer | 15,099 |
| `data/splits/val.csv` | Original split — provenance layer | 3,194 |
| `data/splits/test.csv` | Original split — provenance layer | 3,273 |
| `data/dataset_split/train.csv` | **Canonical — used for all preprocessing and training** | 15,083 |
| `data/dataset_split/val.csv` | **Canonical — used for all preprocessing and training** | 3,191 |
| `data/dataset_split/test.csv` | **Canonical — used for all preprocessing and training** | 3,270 |

### 6.6 Reproducibility Guarantees

The canonical manifests were generated with a deterministic identity-assignment algorithm (fixed seed `42`, sorted identities). SHA-256 hash stability was verified across multiple regenerations:

- Train canonical SHA-256: `DC01136A51350C8A7D3550A5B2187AAAB9DFF1CE6108EA36CD94E4AF54724FA8`
- Val canonical SHA-256: `23F4423F59E96D99A1AB7A7B7D3ADCAEB5F13E37EC7A7EE29E8D70C2BE686829`
- Test canonical SHA-256: `A1DEEE798316E82BF345605008BE2B55A15C35BE1DD1D8CBC199BEF0C69D1BDE`

These hashes can be used to verify dataset pipeline integrity if the scripts are re-run.

---

---

## 7. Data Splitting and Leakage Prevention

### 7.1 The Identity Leakage Problem

In deepfake detection, a naive random per-video split would almost certainly place videos of the same subject (identity) in both the training and test sets. Because deepfake generation artifacts can be identity-specific — for example, the same face may be used as the manipulation source across many videos — a model could learn to recognize a particular identity's visual characteristics rather than generalising to detect synthetic manipulation. This would produce inflated test metrics that do not reflect true generalization.

The project prevents this by splitting the dataset at the **identity level**: all videos containing a given source identity are assigned exclusively to one split. The test set contains 75 identities that the model has never encountered during training or validation.

### 7.2 Identity-Based Split Implementation

**Script:** `scripts/create_identity_split.py` (archived; canonical manifests are the active output)

**Algorithm:**
1. Load all 21,566 rows from `meta_data.csv`.
2. Extract the 500 unique values from the `source` identity column.
3. Sort identities deterministically (alphabetical sort ensures reproducibility independent of filesystem ordering).
4. Assign identities to splits using fixed seed `42`: first 350 → train, next 75 → validation, final 75 → test.
5. Merge identity assignments back to the metadata rows.

**Verification checks run after generation:**
- Zero identity overlap across train/val, train/test, and val/test.
- All 500 identities assigned exactly once.
- All 21,566 rows covered.
- Zero cross-split duplicate sample paths.
- All referenced files physically exist on disk.
- SHA-256 hashes identical across regenerations (seed=42 is deterministic).

### 7.3 Canonical Manifest Layer

The initial split CSVs (`data/splits/`) preserve all 21,566 rows including the 44 duplicate rows (22 physical videos × 2 method annotations). These are the **provenance layer** — they record all original metadata.

The canonical manifests (`data/dataset_split/`) contain **one row per physical video** (21,544 rows total). These are the files used by all preprocessing scripts and training/evaluation dataloaders. The deduplication policy:

- For the 22 duplicated paths: one canonical row is kept; both method values are stored in `method_annotations`; both original rows are preserved in a JSON `metadata_provenance` column.
- No label conflicts existed in any duplicate group, so deduplication does not affect ground-truth labels.

All downstream operations — frame extraction, spectrogram generation, training, evaluation — consume the canonical manifests exclusively.

### 7.4 Split Statistics

| Split | Identities | Physical Videos | Real Videos | Fake Videos | Class Imbalance |
|---|---|---|---|---|---|
| Train | 350 | 15,083 | 700 (4.64%) | 14,383 (95.36%) | ~1:20.5 |
| Validation | 75 | 3,191 | 150 (4.70%) | 3,041 (95.30%) | ~1:20.3 |
| Test | 75 | 3,270 | 150 (4.59%) | 3,120 (95.41%) | ~1:20.8 |

The class imbalance ratio (~1:20 Real:Fake) is consistent across all three splits, reflecting the dataset's natural composition. This imbalance is significant and would cause a naïve model to converge to predicting Fake for all inputs with artificially high accuracy. It is addressed in the training pipeline — see Section 10.2.

### 7.5 Test Set Integrity

The test split of 3,270 videos is the **held-out evaluation set**. It was not used during any training, validation, hyperparameter selection, or architectural decision. The only time test data was accessed was during the final formal evaluation (Phase 15 / `scripts/evaluate_test.py`) and Grad-CAM generation (Phase 16 / `scripts/generate_gradcam.py`). The demo samples in `testing_data/` were sourced from the test set but are used only for Streamlit demonstration, not for any training or metric computation.

---

## 8. Visual Preprocessing

### 8.1 Overview

Visual preprocessing involves two distinct phases:
1. **Offline preprocessing** (`scripts/preprocess_dataset_offline.py`): runs once over the entire dataset to extract and save raw face-cropped frames to `data/processed_frames/` as `.npy` files.
2. **Online/dynamic transforms** (`preprocessing/video_preprocessing.py`, `VisualTransform`): applied at dataset load time in `__getitem__` — augmentation and normalization are applied dynamically, not stored.

The Streamlit application uses a subset of these same preprocessing classes (`VideoPreprocessor`, `RetinaFaceCropper`, `VisualTransform`) for live inference, ensuring inference-time preprocessing matches training-time preprocessing.

### 8.2 Frame Sampling — `UniformTemporalSampler`

**Class:** `preprocessing.video_preprocessing.UniformTemporalSampler`  
**Config:** `num_frames=16`

**Algorithm:**
1. Opens the video with OpenCV and reads total frame count.
2. Computes 16 evenly-spaced indices using `np.linspace(0, total_frames-1, 16, dtype=int)`.
3. Reads exactly those frames, duplicating indices if the video is shorter than 16 frames.
4. If a video has fewer frames than 16, `linspace` automatically handles it by repeating indices.

**Design rationale:** Uniform temporal sampling ensures consistent coverage of the entire video regardless of duration, avoids bias towards the beginning or end, and is fully deterministic — given the same video, the same 16 frame indices are always selected.

### 8.3 Face Detection and Cropping — `RetinaFaceCropper`

**Class:** `preprocessing.video_preprocessing.RetinaFaceCropper`  
**Backend:** InsightFace `FaceAnalysis` using the `det_10g` model  
**Config:** `margin=0.20`, `target_size=(224, 224)`, `det_size=(640, 640)`  
**Providers:** `['CUDAExecutionProvider', 'CPUExecutionProvider']` (GPU preferred)

**`crop_face(frame)` algorithm:**
1. Convert frame from RGB to BGR (InsightFace uses BGR).
2. Run `FaceAnalysis.get(frame_bgr)` — returns a list of detected faces.
3. If no faces detected: return `None` (triggers fallback logic).
4. If multiple faces detected: select the **largest face by bounding-box area**.
5. Expand the detected bounding box by `margin=20%` in each direction (clamped to image boundaries).
6. Crop the expanded region and resize to `224×224` using `cv2.INTER_CUBIC`.
7. Return the crop and metadata dict containing both the expanded and original bounding boxes.

**`apply_bbox_and_crop(frame, bbox)` — forward-fill method:**
Applies a previously computed bounding box (from an earlier frame) to a new frame. Applies the same margin expansion and resize. Used when direct detection fails but a prior bounding box is available.

**Why 20% margin:** The margin ensures that facial context (hairline, jaw) is included in the crop, which may carry deepfake artifacts, and provides robustness against small bounding-box drift between frames.

**Why largest face:** In videos with multiple visible people, the primary subject is typically the largest face. Selecting the largest is a simple heuristic that avoids the need for identity tracking.

### 8.4 Fallback Hierarchy

The offline preprocessing script and the `VideoPreprocessor` both implement a three-tier fallback:

```
For each of the 16 sampled frames:
│
├── 1. Direct RetinaFace detection
│       → Success: crop face, update last_valid_bbox
│
├── 2. Forward-fill (if last_valid_bbox exists)
│       → Apply previous bbox to current frame
│       → Records as "forward_filled" in failure log
│
└── 3. Center-crop fallback (if no valid bbox ever seen)
        → Extract 224×224 centered region from frame
        → Resize if extracted region is smaller than 224×224
        → Records as "center_fallback" in failure log
        → Last resort: if even center crop has 0 pixels, uses
          a black 224×224 zero array (edge case)
```

**Forward-fill rationale:** Validated in Phase 1.8 to have mean geometric drift of only 1.63% across the 50-video quality check. The 20% bounding-box margin absorbs this drift comfortably. The forward-fill nearly halved the center-crop fallback rate.

**Center-crop fallback:** Produces the worst-quality crops (unaligned to the face) but guarantees the pipeline never crashes and always produces a complete 16-frame array. The model must tolerate these imperfect crops.

**Backward-filling was explicitly excluded:** It was considered in Phase 1.8 but excluded from the frozen methodology to maintain processing simplicity and avoid introducing temporal causality issues in the offline preprocessor.

### 8.5 Augmentation and Normalization — `VisualTransform`

**Class:** `preprocessing.video_preprocessing.VisualTransform`

**Training transforms** (`is_train=True`):
| Transform | Parameters | Purpose |
|---|---|---|
| `HorizontalFlip` | p=0.5 | Geometric augmentation — mirror invariance |
| `Rotate` | ±10°, p=1.0, INTER_CUBIC | Geometric augmentation — small rotation invariance |
| `ColorJitter` | brightness/contrast/saturation=0.1, hue=0.05, p=1.0 | Photometric augmentation |
| `CoarseDropout` | max 1 hole, 2–22px, p=0.1 | Occlusion regularization |
| `Normalize` | ImageNet mean/std | Standard normalization |
| `ToTensorV2` | — | Convert to PyTorch tensor |

**Validation/Test/Inference transforms** (`is_train=False`):
| Transform | Parameters |
|---|---|
| `Normalize` | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| `ToTensorV2` | — |

**Important:** The offline preprocessing script uses `is_train=False` (no augmentation) and stores raw uint8 crops. Augmentations are applied dynamically in the dataset `__getitem__`. This means each training epoch applies *different random augmentations* to the same stored crops, providing effective data augmentation without storing multiple augmented copies.

**Temporal consistency:** In the dataset's multi-frame transform pipeline (in `multimodal_dataset.py`), all 16 frames of a sample share the same random augmentation parameters. This is achieved via `albumentations.Compose` with `additional_targets` mapping `image1` through `image15` to the same transform pipeline. This prevents inconsistency between frames (e.g., some frames flipped while others are not).

### 8.6 Offline Preprocessing Output

**Script:** `scripts/preprocess_dataset_offline.py`  
**Output directory:** `data/processed_frames/`  
**File naming:** Mirrors the dataset directory structure: `data/processed_frames/<category>/<identity>/<filename>.npy`  
**Array format:** `np.ndarray` shape `(16, 224, 224, 3)`, dtype `uint8`, raw unaugmented face crops  
**Skip logic:** If the `.npy` file already exists, the video is skipped — the script is fully resumable.  
**Failure log:** Written to `data/processed_frames/failures.json`

**Full preprocessing statistics:**
| Metric | Value |
|---|---|
| Total videos processed | 21,544 / 21,544 |
| Failures | 0 |
| Total frames extracted | 344,704 |
| Direct detection | 141,166 (41.0%) |
| Forward-fill | 83,946 (24.4%) |
| Center-crop fallback | 119,592 (34.7%) |
| Output size | ~48.33 GB |
| Runtime | ~9h 12m |

### 8.7 Inference-Time Visual Preprocessing (Streamlit)

At inference time in `app.py`, the `VideoPreprocessor` is used with `is_train=False`:
- For video uploads: `VideoPreprocessor.process(video_path, return_raw_crops=True)` → then `VisualTransform(is_train=False).apply(crop)` per frame.
- For image uploads: `RetinaFaceCropper.crop_face(img_np)` → `VisualTransform(is_train=False).apply(crop)`.
- **Fallback in app.py:** If face detection fails on an uploaded image (e.g., `real_image_3120.jpg`), the application falls back to resizing the entire image to 224×224 rather than returning an error, ensuring graceful degradation for demo purposes.

---

## 9. Audio Preprocessing

### 9.1 Overview

Audio preprocessing also runs in two phases:
1. **Offline extraction and spectrogram generation** (`scripts/preprocess_audio_offline.py`): runs once to extract WAV files and generate log-mel spectrogram `.npy` arrays for all 21,544 videos.
2. **Online spectrogram generation** (in `app.py`): the `AudioExtractor` and `SpectrogramGenerator` are called live on uploaded files during Streamlit inference.

### 9.2 Audio Extraction — `AudioExtractor`

**Class:** `preprocessing.audio_preprocessing.AudioExtractor`  
**Dependency:** `imageio-ffmpeg` (provides a bundled FFmpeg binary; no system FFmpeg installation required)

**Constructor:** `AudioExtractor(sample_rate=16000, channels=1)` — locates the FFmpeg executable from `imageio_ffmpeg.get_ffmpeg_exe()`.

**`extract(video_path, output_wav_path)` algorithm:**
1. Validates that the source video file exists.
2. Builds an FFmpeg command:
   - `-vn`: discard video stream
   - `-acodec pcm_s16le`: 16-bit PCM encoding
   - `-ar 16000`: resample to 16 kHz
   - `-ac 1`: downmix to mono
3. Runs FFmpeg via `subprocess.run(..., check=True)`, suppressing terminal output.
4. If FFmpeg fails: removes any partial output file and raises `RuntimeError`.

**Output format:** 16 kHz, mono, 16-bit signed PCM WAV. Duration matches the original video duration (not truncated).

**Why 16 kHz mono:** 16 kHz captures all speech-relevant frequencies (below 8 kHz by Nyquist) while keeping file sizes manageable. Mono reduces data dimensionality without losing the information relevant to deepfake detection, as voice synthesis artifacts are not stereo-specific.

### 9.3 Spectrogram Generation — `SpectrogramGenerator`

**Class:** `preprocessing.audio_preprocessing.SpectrogramGenerator`  
**Output:** `float32` tensor of shape `(3, 224, 224)` stored as `.npy`

**Constructor:** `SpectrogramGenerator(sample_rate=16000, n_mels=128, target_size=224)`

Configures:
- `torchaudio.transforms.MelSpectrogram`: `n_fft=1024`, `hop_length=512`, `n_mels=128`, `f_min=20 Hz`, `f_max=8000 Hz`
- `torchaudio.transforms.AmplitudeToDB`: converts power spectrogram to dB scale

**`generate(wav_path)` algorithm:**
1. Load WAV with `torchaudio.load` → `(channels, samples)` waveform tensor.
2. Resample to 16 kHz if needed (the extraction step enforces this, but a guard is included).
3. Downmix to mono if multi-channel (mean across channel dimension).
4. Apply `MelSpectrogram` → `(1, 128, time_frames)` power spectrogram.
5. Apply `AmplitudeToDB` → log-mel spectrogram in dB.
6. Unsqueeze to `(1, 1, 128, time_frames)`.
7. Bilinear interpolate to `(1, 1, 224, 224)` — fixed spatial size regardless of audio duration.
8. Squeeze to `(1, 224, 224)`.
9. Repeat across 3 channels → `(3, 224, 224)` float32.

**Why `(3, 224, 224)`:** EfficientNet-B0 expects a 3-channel 224×224 image. By treating the log-mel spectrogram as a grayscale image and replicating it to 3 channels, the pretrained backbone can process it directly without architectural modifications.

**Why bilinear resize to fixed 224×224:** Audio duration varies across the dataset. Bilinear interpolation normalizes all spectrograms to the same spatial dimensions, making batch processing straightforward. The time axis is compressed or stretched to fit 224 pixels; frequency resolution (128 mel bins) is also resampled to 224 pixels.

**Why log-mel (dB scale):** Human perception of audio is approximately logarithmic in both frequency and amplitude. The log-mel spectrogram compresses the dynamic range of the power spectrum, making subtle artifacts more prominent and the representation more uniform — both beneficial for neural network learning.

### 9.4 Offline Preprocessing Output

**Script:** `scripts/preprocess_audio_offline.py`  
**Output directories:**
- `data/processed_audio/raw_wav/` — extracted WAV files
- `data/processed_audio/spectrograms/` — `.npy` spectrogram tensors

**File naming:** Mirrors dataset structure: `data/processed_audio/spectrograms/<category>/<identity>/<filename>.npy`

**Full preprocessing statistics:**
| Metric | Value |
|---|---|
| Total videos processed | 21,544 / 21,544 |
| Failures | 0 |
| Skipped | 0 |
| Output size (total) | ~15.51 GB |
| Raw WAVs | ~3.43 GB |
| Spectrograms | ~12.08 GB |
| Runtime | ~46m 14s (~7.77 videos/sec) |

**Skip logic:** Both raw WAV and spectrogram files are checked — if both already exist, the video is skipped. The script is fully resumable.

### 9.5 Dependency Notes

| Dependency | Purpose | Install note |
|---|---|---|
| `imageio-ffmpeg` | Bundled FFmpeg binary for audio extraction | Avoids system PATH dependency |
| `soundfile` | torchaudio WAV backend on Windows | Must be explicitly installed: `pip install soundfile` |
| `torchaudio` | MelSpectrogram and file loading | Part of PyTorch ecosystem |

**Windows-specific issue:** On the initial audio preprocessing test, torchaudio could not load WAV files due to a missing `soundfile` backend. The error manifested as torchaudio backend load warnings. Fixed by `pip install soundfile`.

---

---

## 10. Dataset/DataLoader Implementation

### 10.1 MultimodalDeepfakeDataset

**File:** `dataset/multimodal_dataset.py`  
**Class:** `MultimodalDeepfakeDataset(Dataset)`

**Purpose:** A PyTorch `Dataset` that lazily loads preprocessed `.npy` video frame arrays and audio spectrogram arrays for a given split, applies temporally consistent augmentation, parses three separate labels, and returns a five-element tuple per sample.

**Constructor:** `MultimodalDeepfakeDataset(csv_path, video_dir, audio_dir, is_train=False)`
- Reads the canonical CSV manifest (one row per video).
- Constructs an `albumentations.Compose` pipeline with `additional_targets` — keys `image1` through `image15` all declared as type `'image'` — so all 16 frames receive identical random transforms per sample.

**`__getitem__(idx)` algorithm:**
1. Read `sample_path` from the DataFrame row.
2. Derive `.npy` paths: `video_dir/<base_path>.npy` and `audio_dir/<base_path>.npy`. Raises `FileNotFoundError` if either is missing.
3. Load video: `np.load(video_path)` → `(16, 224, 224, 3)` uint8 array.
4. Build albumentations kwargs: `{'image': frame_0, 'image1': frame_1, ..., 'image15': frame_15}`.
5. Apply the transform pipeline (augmentation + normalization) once — all 16 frames share the same random seed for this call.
6. Reconstruct as `torch.stack([transformed['image'], ...])` → `(16, 3, 224, 224)` float32.
7. Load audio: `np.load(audio_path)` → `(3, 224, 224)` float32 → `torch.from_numpy().float()`.
8. Parse all three labels via `_parse_label`:
   - `video_label` (column `video_label`) → Image Head target
   - `audio_label` (column `audio_label`) → Audio Head target
   - `overall_label` (column `label`) → Fusion Head target
   - Mapping: `'real' → 0.0`, `'fake' → 1.0` (case-insensitive, stripped)
9. Return `(video_tensor, audio_tensor, video_label_tensor, audio_label_tensor, overall_label_tensor)`.

**Training vs. Validation/Test augmentation:**

| Mode | Transforms applied |
|---|---|
| `is_train=True` | HorizontalFlip (p=0.5) + Rotate ±10° (p=0.5) + ColorJitter (p=0.5) + Normalize + ToTensorV2 |
| `is_train=False` | Normalize + ToTensorV2 only |

No augmentation is ever applied at eval time — this is critical for reproducible validation and test metric computation.

### 10.2 Class Imbalance Handling

**Problem:** The training split has a severe class imbalance — approximately 4.64% Real (700) vs 95.36% Fake (14,383). A naïve DataLoader would produce batches that are ~95% Fake, which would allow a model to achieve high accuracy by predicting Fake for everything, and would starve the Real class of gradient signal.

**Solution — `WeightedRandomSampler`:**

The factory computes per-sample sampling weights as the inverse of their class frequency:
- `weight_real = 1.0 / 700` (each Real sample is assigned this weight)
- `weight_fake = 1.0 / 14383` (each Fake sample is assigned this weight)

The `WeightedRandomSampler` draws `len(dataset)` samples with replacement, using these weights. The result is that each training batch contains approximately equal numbers of Real and Fake samples (~50/50), regardless of the dataset's natural distribution.

**Key implementation details:**
- `num_samples=len(dataset)` — the epoch length (number of batches) is preserved; the sampler does not artificially inflate or shrink the epoch.
- `replacement=True` — Real samples are drawn multiple times per epoch (oversampling); Fake samples are drawn less often than their full count (undersampling).
- The sampler applies only to the training DataLoader. Validation and test use `shuffle=False`, sequential loading — the natural unbalanced distribution is preserved, which is required for realistic performance estimation.

**Complementary — `BinaryFocalLoss`:** The loss function provides a second level of imbalance handling by down-weighting easy-to-classify examples (which are disproportionately Fake), reducing the effective dominance of the majority class even within balanced batches.

### 10.3 DataLoader Configuration

**File:** `dataset/dataloader_factory.py`  
**Entry point:** `create_dataloaders(csv_dir, video_dir, audio_dir, batch_size, num_workers)`

| Setting | Train | Val | Test |
|---|---|---|---|
| `batch_size` | 4 | 4 | 8 (in evaluation script) |
| `sampler` | `WeightedRandomSampler` | None | None |
| `shuffle` | N/A (sampler overrides) | False | False |
| `num_workers` | 4 | 4 | 4 |
| `pin_memory` | True | True | True |
| `drop_last` | True | False | False |

`drop_last=True` for training discards the final incomplete batch per epoch, ensuring consistent batch sizes with the AMP-enabled training loop.

`pin_memory=True` enables faster CPU→GPU memory transfers by using pinned (page-locked) memory for DataLoader output tensors.

---

## 11. Loss Function and Training Strategy

### 11.1 BinaryFocalLoss

**File:** `training/losses.py`  
**Class:** `BinaryFocalLoss(gamma=2.0, reduction='mean')`

Focal Loss was introduced to address extreme class imbalance in object detection (Lin et al., 2017) and is directly applicable here. It modifies standard Binary Cross-Entropy by multiplying each sample's loss by a modulating factor `(1 - p_t)^gamma`, where `p_t` is the model's predicted probability for the correct class.

**Mathematical formulation:**
```
BCE(logit, target) = -[target * log(sigmoid(logit)) + (1-target) * log(1-sigmoid(logit))]

p_t = sigmoid(logit)     if target == 1
    = 1 - sigmoid(logit)  if target == 0

Focal Weight = (1 - p_t)^gamma

Loss = Focal_Weight * BCE(logit, target)
```

**Implementation detail:** The raw BCE term is computed using `F.binary_cross_entropy_with_logits(logits, targets, reduction='none')` — this is numerically stable because it combines the sigmoid and cross-entropy in a single stable log-sum-exp operation rather than computing `sigmoid(logit)` and then taking `log`. The sigmoid for the focal weight computation is calculated separately.

**Effect of `gamma=2.0`:**
- If the model is highly confident and correct (p_t ≈ 1): weight ≈ `(1-1)^2 = 0` → near-zero loss (easy example down-weighted)
- If the model is uncertain or wrong (p_t ≈ 0): weight ≈ `(1-0)^2 = 1` → full BCE loss preserved (hard example fully penalized)
- This focuses gradient on genuinely difficult or misclassified examples, preventing the loss from being dominated by the many easy Fake predictions.

### 11.2 MultiTaskFocalLoss

**File:** `training/losses.py`  
**Class:** `MultiTaskFocalLoss(gamma=2.0)`

Wraps `BinaryFocalLoss` and applies it independently to all three prediction heads.

**Forward signature:** `forward(preds: dict, video_label, audio_label, overall_label) -> dict`

**Loss computation:**
```
fusion_loss = BinaryFocalLoss(preds['fusion'], overall_label)
audio_loss  = BinaryFocalLoss(preds['audio'], audio_label)

# Image head produces (B*T, 1) frame-level logits.
# video_label is (B, 1). Expand via repeat_interleave:
expanded_label = video_label.repeat_interleave(T, dim=0)  # (B*T, 1)
image_loss = BinaryFocalLoss(preds['image'], expanded_label)

total_loss = image_loss + audio_loss + fusion_loss
```

**Returns:** `{'total': ..., 'image': ..., 'audio': ..., 'fusion': ...}`

**Design choice — equal unweighted summation:** All three head losses contribute equally to the total. No loss weighting was applied. The rationale is that the project's primary goal is modality decoupling, not maximizing a single head's accuracy. Equal weighting ensures no head is systematically privileged.

**`repeat_interleave` explanation:** During the forward pass with `return_all=True`, the visual encoder returns `(B×T, 1280)` unpooled frame features, and the image head produces `(B×T, 1)` per-frame logits. The `video_label` is `(B, 1)` — one label per video, not per frame. `repeat_interleave(T, dim=0)` expands `[label_vid1, label_vid2]` to `[label_vid1, label_vid1, ...(×T), label_vid2, label_vid2, ...(×T)]`, correctly matching each frame's logit to its video's label. This avoids creating a separate per-frame label dataset.

### 11.3 Optimizer and Scheduler

**Optimizer:** `AdamW(lr=1e-4, weight_decay=1e-4)`
- AdamW decouples weight decay from the gradient update step, which produces better regularization than the original Adam with L2 penalty.
- `lr=1e-4` is the standard recommended starting rate for fine-tuning pretrained EfficientNet models.
- `weight_decay=1e-4` provides moderate L2 regularization.

**Scheduler:** `ReduceLROnPlateau(mode='min', patience=2, factor=0.5)`
- Monitors total validation loss.
- If validation loss does not improve for 2 consecutive epochs, the learning rate is reduced by 50%.
- This allows the model to explore efficiently at a higher learning rate early in training and fine-tune at lower rates if it plateaus.

**Note:** The V1 architecture used `CosineAnnealingLR`. The switch to `ReduceLROnPlateau` in V2 was made to produce adaptive decay conditioned on actual validation performance rather than a fixed schedule.

### 11.4 Automatic Mixed Precision (AMP)

`torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast` are used throughout training and evaluation. AMP reduces memory usage by performing forward passes in `float16` where precision is not critical, while maintaining `float32` precision for the loss and gradient computations. The `GradScaler` scales the loss before the backward pass and unscales gradients before the optimizer step to prevent numerical underflow in float16.

**Observed VRAM usage (RTX 4050, batch_size=4):** 3.03 GB allocated / 3.27 GB reserved (out of 6 GB).

### 11.5 Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` is applied after `scaler.unscale_()` and before `scaler.step()`. This prevents gradient explosions that can destabilize training, particularly relevant in multi-task settings where gradients from three separate loss terms accumulate in the shared backbone parameters.

---

## 12. Training and Checkpoint Selection

### 12.1 Training Entry Point

**Script:** `scripts/train.py`  
**Command:** `.\.venv_gpu\Scripts\python.exe scripts\train.py`

The script configures all hyperparameters, instantiates the DataLoaders, model, and Trainer, then runs a 10-epoch loop printing per-epoch summaries. All epoch history is logged to `results/performance_curves/training_history.json`.

**Final configuration used:**

| Hyperparameter | Value |
|---|---|
| `batch_size` | 4 |
| `num_epochs` | 10 |
| `learning_rate` | 1e-4 |
| `weight_decay` | 1e-4 |
| `focal_gamma` | 2.0 |
| `gradient_clip` | 1.0 |
| `num_workers` | 4 |
| `pretrained` | True (ImageNet weights) |
| `dropout` | 0.3 |

### 12.2 Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6 GB |
| CUDA | 11.8 |
| Peak VRAM (training) | 3.03 GB allocated / 3.27 GB reserved |

### 12.3 Complete Epoch-Level Training History

All values sourced directly from `results/performance_curves/training_history.json`.

| Epoch | Train Total | Train Img | Train Aud | Train Fus | Val Total | Val Img | Val Aud | Val Fus | Val Fus Acc | Val Aud Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0743 | 0.0337 | 0.0170 | 0.0236 | 0.0258 | 0.0137 | 0.0005 | 0.0116 | 99.81% | 99.87% |
| 2 | 0.0187 | 0.0096 | 0.0029 | 0.0061 | 0.0353 | 0.0196 | 0.0055 | 0.0102 | 99.53% | 99.53% |
| 3 | 0.0147 | 0.0076 | 0.0028 | 0.0044 | 0.0339 | 0.0192 | 0.0007 | 0.0141 | 99.69% | 99.97% |
| 4 | 0.0100 | 0.0051 | 0.0015 | 0.0034 | 0.0223 | 0.0120 | 0.0002 | 0.0101 | 99.81% | 99.97% |
| 5 | 0.0111 | 0.0056 | 0.0021 | 0.0034 | 0.0211 | 0.0115 | 0.0001 | 0.0095 | 99.72% | 100.00% |
| **6** | **0.0088** | **0.0039** | **0.0022** | **0.0027** | **0.0189** | **0.0096** | **0.0000** | **0.0093** | **99.81%** | **100.00%** |
| 7 | 0.0079 | 0.0039 | 0.0013 | 0.0026 | 0.0292 | 0.0171 | 0.0000 | 0.0121 | 99.50% | 100.00% |
| 8 | 0.0065 | 0.0029 | 0.0010 | 0.0026 | 0.0452 | 0.0255 | 0.0001 | 0.0196 | 99.00% | 100.00% |
| 9 | 0.0083 | 0.0041 | 0.0008 | 0.0034 | 0.0784 | 0.0559 | 0.0001 | 0.0225 | 98.40% | 100.00% |
| 10 | 0.0042 | 0.0020 | 0.0006 | 0.0016 | 0.0697 | 0.0359 | 0.0000 | 0.0338 | 97.77% | 100.00% |

**Bold row = best checkpoint (Epoch 6).**

### 12.4 Training Dynamics Observations

**Rapid audio convergence:** The audio head loss collapsed to near-zero by Epoch 6 (`val audio_loss = 0.0000`) and the Audio Head achieved 100% validation accuracy. Audio deepfake detection is an easier signal for the model to learn, likely because GAN/synthesis artifacts in the audio spectrogram are more distinctive than visual artifacts. Audio accuracy remained at 100% for all epochs 6–10.

**Image head drives overfitting:** From Epoch 7 onwards, the total validation loss increased despite the training loss continuing to decrease. Inspecting per-head losses reveals that the image loss on validation grew substantially (0.0096 at Epoch 6 → 0.0559 at Epoch 9), while the audio and fusion heads remained stable. The image head learned to overfit to the training distribution of visual artifacts after Epoch 6.

**Fusion head follows image head:** The fusion head's validation loss also increased post-Epoch 6, which is expected since it concatenates visual and audio features — the visual overfitting propagates into the fused representation.

**Conclusion — Epoch 6 is the correct best checkpoint:** Epoch 6 represents the optimal bias-variance tradeoff across all three heads simultaneously: the audio head had converged to near-perfect accuracy, the image head had not yet overfit, and the fusion head was at its lowest validation loss.

### 12.5 Checkpoint Selection and Saving

**Best checkpoint criterion:** Saved whenever total validation loss is strictly lower than the previous best. The best checkpoint across all 10 epochs is **Epoch 6** (`val total_loss = 0.0189`).

**Checkpoint file:** `checkpoints/best_model.pt`  
**Latest checkpoint file:** `checkpoints/latest_model.pt` (overwritten every epoch)

**Checkpoint contents:**
```python
{
    'epoch': 6,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_metrics': { ... },   # per-head val metrics at this epoch
    'model_config': {'pretrained': False}
}
```

**Loading at evaluation/inference time:**
```python
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
model = MultiHeadDeepfakeModel(pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

`weights_only=False` is required because the checkpoint contains non-tensor Python objects. `pretrained=False` is correct at load time — the trained weights from the checkpoint override the backbone initialization.

---

## 13. Evaluation Methodology

### 13.1 Evaluation Entry Point

**Script:** `scripts/evaluate_test.py`  
**Command:** `.\.venv_gpu\Scripts\python.exe scripts\evaluate_test.py`

The evaluation script runs a single inference pass over the entire held-out test set (3,270 samples) using the `best_model.pt` checkpoint. No gradient computation is performed (`torch.no_grad()`). AMP is used for consistent numerical behavior with training.

### 13.2 Image Head Probability Aggregation

During evaluation, the image head produces `(B×16, 1)` per-frame logits for each batch. These are converted to per-frame probabilities and averaged across the 16 frames to yield one image-head probability per video:

```python
img_probs = torch.sigmoid(img_logits).view(B, 16).mean(dim=1)  # (B,)
```

This mirrors the implicit aggregation that the model learns during training (where the mean-pooled `(B, 1280)` video representation feeds the fusion head). At evaluation, frame-level logits are returned directly and averaged post-sigmoid to produce a stable per-video probability estimate.

### 13.3 Classification Threshold

All three heads use a threshold of `0.5`:
- `img_pred = (img_prob > 0.5).astype(float)`
- `aud_pred = (aud_prob > 0.5).astype(float)`
- `fus_pred = (fus_prob > 0.5).astype(float)`

No threshold tuning was performed. The default 0.5 threshold is used throughout.

### 13.4 Metrics Computed

For each of the three prediction heads independently, the following metrics are computed against the respective ground-truth labels:

| Metric | Description | Implementation |
|---|---|---|
| **Accuracy** | Overall correct classification rate | `sklearn.metrics.accuracy_score` |
| **Precision** | TP / (TP + FP) | `precision_score(zero_division=0)` |
| **Recall** | TP / (TP + FN) | `recall_score(zero_division=0)` |
| **F1 Score** | Harmonic mean of Precision and Recall | `f1_score(zero_division=0)` |
| **ROC-AUC** | Area under the ROC curve (threshold-independent) | `roc_auc_score` |

For the Image Head, ground-truth labels are `video_label` (`Fake` if the video was visually manipulated). For the Audio Head, ground-truth labels are `audio_label`. For the Fusion Head, ground-truth labels are `overall_label` (same as `video_label`).

### 13.5 Four-Category Modality Diagnostic

The evaluation script additionally computes per-category accuracy and mean fake probabilities for all three heads, stratified by the four `type` categories (`RealVideo-RealAudio`, `FakeVideo-RealAudio`, `RealVideo-FakeAudio`, `FakeVideo-FakeAudio`). This diagnostic is the core scientific evidence for modality decoupling and is documented in full in Section 14.

### 13.6 Outputs Generated

| Output | Location |
|---|---|
| Global per-head metrics JSON | `results/metrics/test_metrics.json` |
| Per-sample predictions + probabilities | `results/predictions/test_predictions.csv` |
| Image/Audio/Fusion confusion matrices | `results/confusion_matrices/*.png` |
| Image/Audio/Fusion ROC curves | `results/roc_curves/*.png` |
| Four-category modality diagnostic CSV | `results/modality_analysis/modality_category_results.csv` |

All outputs are deterministic — re-running the script with the same checkpoint and test set will produce identical results.

---
