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
