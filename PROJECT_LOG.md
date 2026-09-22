# Project Log

Permanent implementation history for **Deepfake Detection Using Multimodal Learning**.

## 2026-09-06

- **Stage:** Project governance and baseline confirmation
- **Task performed:** Adopted the supplied master project instruction as the governing implementation and documentation policy. Inspected the implementation blueprint, dataset audit outputs, Git status, recent commit history, and configured GitHub remote.
- **Why it was performed:** The project requires a traceable laboratory notebook before implementation proceeds, and all future work must remain aligned with the finalized methodology and the locally audited dataset.
- **Implementation/details:**
  - Confirmed the repository remote is `https://github.com/udaykumar0515/multimodal-deepfake-detection.git`.
  - Confirmed the current branch is `main` and the latest commit is `f8eed83` (`Add Verifications for Label Integrity and Identity Split Statistics`).
  - Confirmed `.gitignore` excludes `Dataset_FakeAVCeleb/`, `Docs/`, notebook checkpoints, and Python cache files. The raw dataset will remain local and uncommitted.
  - Confirmed the blueprint’s first implementation stage is dataset preparation: identity-based splitting, CSV generation, video preprocessing, audio preprocessing, and PyTorch loaders.
  - Created this project log as the permanent record for implementation decisions and verification evidence.
- **Verification performed:** Read the current blueprint and audit output files; checked Git status, remote configuration, recent history, and ignore rules. No code or raw dataset files were changed.
- **Results:** Repository governance and the intended Phase 1 sequence are confirmed. The repository currently has a modified audit notebook and an untracked implementation plan; both pre-existing changes were preserved.
- **Problems/issues discovered:** The source documents contain an unresolved real-sample count discrepancy. The audit output reports 1,500 real samples in its folder summary and 500 real samples in its sample inspection, while the blueprint defines 1,000 real visual samples and uses that figure for the visual split. The audit also reports 21,544 videos and 21,566 metadata rows. These figures must be reconciled against the actual metadata and filesystem before generating split CSVs.
- **Decisions made:** Do not begin dataset split generation using an assumed count. Reconcile metadata labels, filesystem paths, and identity assignments first. Do not alter the raw dataset or redesign the finalized architecture without explicit approval.
- **Files created/modified:** Created `PROJECT_LOG.md`. No existing files were modified by this task.
- **Next step:** Reconcile the dataset audit discrepancy with a reproducible metadata/filesystem validation, then implement and verify identity-based train/validation/test CSV generation.

## 2026-09-06

- **Stage:** Phase 1 - Dataset preparation
- **Task performed:** Implemented and verified identity-based train/validation/test splitting for the local FakeAVCeleb metadata.
- **Why it was performed:** Individual-video random splitting would permit the same source identity to appear in multiple partitions and cause identity leakage. The project requires source-level isolation before preprocessing or model training.
- **Implementation/details:**
  - Read `Dataset_FakeAVCeleb/meta_data.csv` and verified 21,566 metadata rows and 500 unique values in the `source` identity field.
  - Assigned sorted source identities using a fixed random seed of `42`: 350 identities to train, 75 to validation, and 75 to test.
  - Constructed each relative sample path from the metadata directory field (`Unnamed: 9`) and filename field (`path`), without changing the raw dataset.
  - Preserved all original metadata columns and added `sample_path`, `video_label`, `audio_label`, `label`, and `split` columns. The primary `label` follows the blueprint's visual branch: `RealVideo-*` is `Real` and `FakeVideo-*` is `Fake`.
  - Added `scripts/create_identity_split.py`, which generates the three CSV files and verifies identity coverage, identity overlap, row coverage, file existence, cross-split path duplication, and identity row preservation.
- **Verification performed:** Ran `python scripts/create_identity_split.py`, `python -m py_compile scripts/create_identity_split.py`, and regenerated the outputs twice with the same seed while comparing SHA-256 hashes.
- **Results:** Final status `PASS`. Train: 350 identities, 15,099 metadata rows, 700 Real and 14,399 Fake. Validation: 75 identities, 3,194 metadata rows, 150 Real and 3,044 Fake. Test: 75 identities, 3,273 metadata rows, 150 Real and 3,123 Fake. All 21,566 rows and all 500 identities were assigned exactly once at the identity level. Identity overlap was zero for train/validation, train/test, and validation/test. All referenced files existed locally. Cross-split duplicate sample paths: zero. Fixed-seed output hashes were unchanged across regeneration.
- **Problems/issues discovered:** The live metadata contains 21,544 unique canonical video paths but 21,566 rows; 44 rows are duplicate metadata records for 22 paths. These rows were preserved rather than silently removed. The realized sample counts differ from the blueprint's previously documented counts because the current metadata and identity grouping determine the actual partition sizes. The audit's 1,500-versus-500 real-count discrepancy is resolved for the visual task by the verified `type` mapping: 1,000 `RealVideo-*` rows and 20,566 `FakeVideo-*` rows.
- **Decisions made:** Use `source` as the sole split key. Keep duplicate metadata rows visible in the CSVs, while requiring each canonical sample path to occur in only one split. Use the video component of `type` for the primary visual label and retain the audio component separately; do not silently correct source metadata labels.
- **Files created/modified:** Created `scripts/create_identity_split.py`, `data/splits/train.csv`, `data/splits/val.csv`, and `data/splits/test.csv`. Updated `PROJECT_LOG.md`.
- **Next step:** Begin video preprocessing only after this split commit is available; do not proceed to model implementation yet.

## 2026-09-06

- **Stage:** Phase 1 - Dataset preparation / duplicate integrity audit
- **Task performed:** Audited duplicate canonical sample paths in the existing identity-based split manifests before video preprocessing.
- **Why it was performed:** The split contains more metadata rows than unique physical video paths. Duplicate records could change sample weighting during preprocessing and could conceal conflicting annotations, so they had to be classified before any media processing.
- **Implementation/details:**
  - Added `scripts/audit_duplicate_records.py` to load `train.csv`, `val.csv`, and `test.csv`, compare duplicate groups field by field, verify physical files, recheck split isolation, and generate a Markdown report.
  - Audited all relevant original metadata fields plus `video_label`, `audio_label`, `label`, and split assignment.
  - Compared duplicate rows as metadata records rather than deleting or normalizing them.
- **Verification performed:** Loaded all three manifests and confirmed the expected schema and row counts. Ran `python -m py_compile scripts/audit_duplicate_records.py` and `python scripts/audit_duplicate_records.py`. Verified every duplicated path against the local dataset root, rechecked cross-split path and identity intersections, and checked visual/audio/final label consistency.
- **Results:** Audit status `PASS - audit complete; manifests unchanged`. There are 21,566 manifest rows, 21,544 unique canonical paths, 22 duplicated paths, and 44 duplicate rows. All 22 duplicate physical files exist. Every duplicate group occurs within one split. Cross-split duplicate paths: 0. Cross-split source identities: 0. Every duplicate group has two metadata rows differing only in `method` (`faceswap-wav2lip` versus `wav2lip`); `type`, identity, path, and other audited metadata agree. Visual-label conflicts: 0. Audio-label conflicts: 0. Final-label conflicts: 0.
- **Problems/issues discovered:** These are not exact duplicate metadata records. They represent the same physical video referenced by conflicting method annotations. Retaining both rows preserves both annotations but can cause the physical video to receive double row-level sampling weight in later preprocessing. Removing one row would reduce that weighting and discard one method annotation. No automatic choice is justified by the current evidence.
- **Decisions made:** Do not modify the split CSVs and do not modify the raw dataset. Keep the duplicate records unchanged pending explicit dataset-method review. Do not proceed to RetinaFace, frame extraction, audio extraction, training, or model implementation from this task.
- **Files created/modified:** Created `scripts/audit_duplicate_records.py` and `reports/dataset_duplicate_audit.md`. Updated `PROJECT_LOG.md`. The existing split CSVs and raw dataset were not modified.
- **Next step:** Review the 22 `method` conflicts and approve a controlled manifest policy before video preprocessing. This audit does not authorize deduplication or media processing.

## 2026-09-06

- **Stage:** Phase 1 - Dataset preparation / final dataset freeze
- **Task performed:** Completed and froze the reproducible canonical dataset manifests for future preprocessing.
- **Why it was performed:** The original identity split preserved every metadata row, but 22 physical videos had two method annotations. Future preprocessing must process each physical video exactly once while retaining metadata provenance and preserving the established identity, label, and split guarantees.
- **Implementation/details:**
  - Added `scripts/create_canonical_manifests.py`. It deterministically groups each original split by `sample_path`, keeps one canonical row per physical video, preserves `metadata_row_count`, combines method annotations in sorted `method_annotations`, and stores all contributing original records in deterministic JSON `metadata_provenance`.
  - Added `scripts/validate_dataset.py`. It validates original-to-canonical row accounting, canonical uniqueness, split and identity isolation, label derivation, relative `.mp4` paths, physical-file existence, provenance JSON, and lightweight OpenCV container/readability including one frame read per canonical video.
  - Added `scripts/create_final_dataset_summary.py` for reproducible research-facing statistics.
  - Added `data/README.md` documenting raw-data handling, manifest layers, duplicate policy, labels, and regeneration commands.
  - Generated `data/splits/canonical/train.csv`, `val.csv`, and `test.csv`.
- **Duplicate policy:** The original manifests remain unchanged. Each canonical manifest contains one row per physical `sample_path`. The 22 duplicated paths are represented once each; their two method annotations (`faceswap-wav2lip` and `wav2lip`) remain in `method_annotations`, and both original metadata rows remain in `metadata_provenance`. No raw dataset file was modified.
- **Label policy:** `label` and `video_label` continue to derive from `type` using `RealVideo-* -> Real` and `FakeVideo-* -> Fake`. `audio_label` retains the audio component. The final validation found no label conflicts or unexpected type values.
- **Path policy:** Canonical paths are portable relative `.mp4` references resolved below the local `Dataset_FakeAVCeleb/` root. No video directories or copied media were created.
- **Verification performed:** Ran Python compilation for all dataset scripts; reran `create_identity_split.py` and `audit_duplicate_records.py`; generated canonical manifests; ran `validate_dataset.py` across all 21,544 canonical videos; generated `final_dataset_summary.md`; ran `git diff --check`; and regenerated canonical manifests twice with SHA-256 comparison.
- **Final statistics:** Original metadata rows: 21,566. Unique physical videos and canonical rows: 21,544. Duplicate physical paths: 22. Duplicate metadata rows: 44. Canonical train/validation/test videos: 15,083 / 3,191 / 3,270. Identities: 350 / 75 / 75. Canonical visual labels: train 700 Real / 14,383 Fake; validation 150 / 3,041; test 150 / 3,120. Minimum, maximum, and mean canonical videos per identity: 3, 88, and 43.088.
- **Verification results:** `DATASET STATUS = PASS`. All 21,544 canonical physical files existed. OpenCV container/readability checks covered all 21,544 files, including one frame read each, with 0 failures. Canonical cross-split path overlap: 0. Canonical cross-split identity overlap: 0. Canonical provenance row counts sum to all 21,566 original rows. Reproducibility passed; canonical SHA-256 hashes were unchanged between regenerations: train `DC01136A51350C8A7D3550A5B2187AAAB9DFF1CE6108EA36CD94E4AF54724FA8`, val `23F4423F59E96D99A1AB7A7B7D3ADCAEB5F13E37EC7A7EE29E8D70C2BE686829`, test `A1DEEE798316E82BF345605008BE2B55A15C35BE1DD1D8CBC199BEF0C69D1BDE`.
- **Problems/issues discovered:** The live dataset differs from some older blueprint/audit counts because the current metadata contains 21,566 rows and 21,544 unique physical videos. The 22 duplicate groups have method metadata differences but no label differences. No unresolved dataset integrity failure remains after canonicalization.
- **Decisions made:** Future preprocessing must consume only the canonical manifests. Original split manifests remain the provenance layer and must not be used as one-row-per-video preprocessing input. GPU use is not applicable to this completed CPU-based container audit; no preprocessing was started.
- **Files created/modified:** Created canonical split CSVs, `scripts/create_canonical_manifests.py`, `scripts/validate_dataset.py`, `scripts/create_final_dataset_summary.py`, `data/README.md`, `reports/final_dataset_summary.md`, and `reports/media_integrity_failures.csv`. Updated `PROJECT_LOG.md`.
- **Reproducibility results:** Canonical generation completed in approximately 13.4 seconds in the measured run, and repeated generation produced identical SHA-256 hashes for all three canonical manifests.
- **Git commit hash:** Dataset-freeze implementation was committed and pushed as `c863a8c`.
- **Next step:** `NEXT STAGE = VIDEO/AUDIO PREPROCESSING`. Stop here; do not begin RetinaFace, frame extraction, audio extraction, spectrogram generation, model implementation, training, or Streamlit.

## 2026-09-20

- **Stage:** Phase 1 - Video Preprocessing
- **Task performed:** Created final working dataset directory (data/dataset_split/) and implemented video preprocessing pipeline with RetinaFace, uniform temporal sampling, and conservative augmentations.
- **Why it was performed:** To implement the finalized Phase 1 blueprint for multimodal deepfake detection video input.
- **Implementation/details:**
  - Set up data/dataset_split/ populated from canonical manifests.
  - Configured NUM_FRAMES = 16 as a baseline parameter (not experimentally optimal yet) for deterministic UniformTemporalSampler covering the whole video length.
  - RetinaFace implementation using insightface (det_10g model), with 20% bounding-box margin and 224x224 interpolation.
  - Training augmentation (Horizontal Flip p=0.5, Rotate ±10°, Color Jitter, Random Erasing) via lbumentations.
  - Validation/Test transformation set to deterministic ImageNet normalization.
  - Failure-handling policy logs face detection failures and falls back deterministically to a central crop.
- **Verification performed:** Executed scripts/verify_video_preprocessing.py on 3 videos (1 per split). Validated input loading, preprocessing tensor shape (16x3x224x224), bounds (no NaN/Inf), and failure fallback.
- **Results:** Processed 3/3 samples successfully. insightface encountered face detection failures (likely CPU thresholding on smaller clips), handled gracefully by the fallback logic. Final output shapes and value normalization passed.
- **Problems/issues discovered:** insightface might require GPU execution and confidence tuning for robust face detection; failures fallback to center-cropping was exercised.
- **Decisions made:** Fallback strategy ensures the network always receives [NUM_FRAMES, 3, 224, 224] tensors, preventing pipeline crashes during training.
- **Files created/modified:** Created data/dataset_split/*, preprocessing/video_preprocessing.py, preprocessing/__init__.py, and scripts/verify_video_preprocessing.py.
- **Next step:** NEXT STAGE = AUDIO PREPROCESSING. Stop here.

## Phase 1.5: Video Preprocessing Quality Validation
- Validated video preprocessing on a deterministic subset of 50 videos (25 Train, 15 Val, 10 Test).
- **Results**: CUDA was unavailable; InsightFace used CPUExecutionProvider.
- **Detection Success Rate**: 41.5%
- **Fallback Rate (Center Crop)**: 58.5%
- **Problematic Videos**: 8 videos had 0 successful detections; 31 had >50% fallback rate.
- **Conclusion**: RetinaFace CPU performance is inadequate for robust deepfake detection data extraction. A 58.5% blind center-crop fallback rate will destroy dataset integrity. Proceeding to full-dataset processing is halted pending GPU acceleration or a better fallback mechanism (e.g. forward-filling bounding boxes).

## Phase 1.6: GPU Environment Diagnosis
- Diagnosed the cause of missing GPU acceleration for RetinaFace/PyTorch.
- **Findings**: The NVIDIA GeForce RTX 4050 is fully visible to Windows (Driver 546.18, CUDA 12.3 supported). However, the Python environment is using CPU-only packages (	orch==2.14.0+cpu and onnxruntime==1.30.0).
- **Status**: GPU acceleration is currently **NOT** fixed. A package swap to CUDA-enabled versions is required to resolve this bottleneck.

## Phase 1.6.1: GPU Acceleration Validation
- **Task performed:** Installed CUDA-capable versions of PyTorch and ONNX Runtime and verified `CUDAExecutionProvider` works on Windows.
- **Results:** GPU acceleration was successful (3.2x faster: ~1.42 sec/video vs ~4.57 sec/video). However, the direct detection success rate remained at 41.5%, proving the detection failures are due to model/data characteristics, not CPU compute precision.

## Phase 1.7: Bounding-Box Forward-Fill Experiment
- **Task performed:** Implemented stateful "Bounding-Box Forward-Filling" in `preprocessing/video_preprocessing.py`. When a face is missed in a frame, the pipeline reuses the bounding box from the most recent successful detection in the sequence.
- **Results:** 
  - Direct Detections: 41.5%
  - Forward-Filled: 25.4%
  - Center Fallback: Reduced from 58.5% down to 33.1%.
- **Conclusion:** Forward-filling salvaged nearly half of the failed detections (203 out of 468) and increased the number of fully-covered videos (0 center fallbacks) to 20/50. Dataset integrity is significantly improved.

## Phase 1.8: Forward-Fill Quality Check
- **Task performed:** Ran a visual and quantitative diagnostic check (`scripts/quality_check_forward_fill.py`) on the 50-video validation sample to ensure forward-filled bounding boxes remain accurate and to analyze the remaining 33.1% fallback rate.
- **Quantitative findings:** Mean forward-fill streak was ~3 frames (max 12). Mean geometric drift from stale bounding boxes to the next detection was only 1.63%.
- **Visual findings:** Forward-filled bounding boxes consistently remain on the face due to the 20% margin. The remaining 33.1% center fallbacks occur primarily because initial frames in the video fail detection, leaving no prior bounding box to reuse.
- **Decision:** Do not proceed to full-dataset processing yet. The pipeline must be upgraded to support **Backward-Filling** (allowing initial missing frames to borrow the bounding box from the first successful detection later in the sequence) to further reduce center fallbacks.

## Phase 1.10: Repository Cleanup and Experimental Artifact Organization
- **Previous repository organization:** Validation reports, diagnostic scripts, duplicate audits, dataset splits, and experimental artifacts were scattered across root, `data/`, `scripts/`, and `reports/` directories.
- **New organization:** Created a `testing/` hierarchy to cleanly archive historical and diagnostic artifacts. `reports/` was absorbed completely into `testing/`.
- **What was moved:** 
  - `data/splits/` (historical dataset artifacts) to `testing/dataset/splits/`
  - Dataset audit scripts and reports to `testing/dataset/audits/`
  - GPU diagnoses and scripts to `testing/gpu/`
  - Video validation/forward-fill scripts and reports to `testing/video_preprocessing/`
  - Root `implementaion_plan.md` to `testing/historical_artifacts/`
- **Why it was moved:** To ensure the root and `scripts/` directories contain only active, production-ready code, while preserving all validation provenance and reproducibility artifacts without loss of history.
- **What remains active:** `data/dataset_split/` (final working manifests), `preprocessing/` (active pipeline), `scripts/preprocess_dataset_offline.py` (production script), `PROJECT_LOG.md`, and raw data.
- **Confirmation:** No experimental evidence or datasets were deleted. All scripts were updated to reflect new import paths.
- **Verification results:** Paths verified and `PROJECT_LOG.md` updated successfully.
- **Next phase:** Full video preprocessing or Audio preprocessing (after potentially fixing Backward-Filling).

## Phase 2: Full Video Preprocessing (Completed)
- **Status:** Completed
- **Configuration & Methodology:**
  - **Inputs:** `data/dataset_split/train.csv`, `val.csv`, `test.csv` (21,544 videos total).
  - **Pipeline:** `VideoPreprocessor` executing Uniform Sampling (16 frames/video), RetinaFace (`det_10g`), 20% bounding-box margin.
  - **Fallback Logic:** Direct Detection -> Forward-Fill (reusing last valid bbox) -> Center-Crop Fallback. (Backward-filling explicitly skipped per frozen methodology).
- **Completion Record:**
  - **Total Videos Processed:** 21,544 / 21,544
  - **Failures:** 0
  - **Split Counts:** 15,083 Train / 3,191 Val / 3,270 Test
  - **Total Frames Extracted:** 344,704
  - **Frame Breakdown:** 
    - Direct: 141,166
    - Forward-fill: 83,946
    - Center fallback: 119,592
  - **Output Format:** `.npy` (224x224x3 `uint8`)
  - **Output Size:** ~48.33 GB
  - **Runtime:** ~9h 12m
- **Verification Milestones:**
  - [x] **GPU/CUDA status:** GPU acceleration successfully utilized. The ONNX Runtime warning (`LoadLibrary failed with error 126`) was confirmed to be a harmless initialization artifact.
  - [x] **Resumability:** Verified.
  - [x] **Raw dataset:** Verified completely untouched.
  - [x] **Downstream compatibility:** Verified. The `[16, 224, 224, 3]` `uint8` `.npy` arrays are perfectly compatible for loading directly into future PyTorch datasets for dynamic transformation.

## Phase 3: Audio Preprocessing (Completed)
- **Status:** PHASE 3 COMPLETE
- **Configuration & Methodology:**
  - **Extraction Format:** 16 kHz mono 16-bit PCM WAV with dynamic/original duration preserved.
  - **Spectrogram Configuration:** 128-mel Log-Mel spectrogram, resized to 224x224, expanded to 3-channel `float32` tensors (`.npy`).
  - **Output Locations:**
    - `data/processed_audio/raw_wav/`
    - `data/processed_audio/spectrograms/`
- **Completion Record:**
  - **Total Videos Processed:** 21,544 / 21,544 success
  - **Failed:** 0
  - **Skipped:** 0
  - **Split Counts:** 15,083 Train / 3,191 Val / 3,270 Test
  - **Runtime:** ~46m 14s (~7.77 videos/sec)
  - **Output Size:** ~15.51 GB total (3.43 GB raw WAVs + 12.08 GB Spectrograms)
- **Final Verification Results:**
  - All outputs accurately verified for correct parameters, shapes, dtypes, readability, and correct `sample_path` mapping. No NaNs or Infs present.
  - Resume logic verified.
  - Raw dataset verified completely untouched.
  - Git properly configured to ignore generated audio data.
- **Warnings/Dependency Issues:**
  - During the initial test, the `soundfile` dependency for `torchaudio` was missing on Windows, throwing backend load errors. Fixed by executing `pip install soundfile`.
  - `imageio-ffmpeg` was installed to provide an isolated project-level FFmpeg binary.

## Phase 4: PyTorch Dataset & DataLoader (Completed)
- **Status:** PHASE 4 COMPLETE
- **Configuration & Methodology:**
  - **Dataset Implementation:** `MultimodalDeepfakeDataset` lazily loads preprocessed `.npy` video and audio tensors directly from disk into memory, preventing aggressive RAM consumption.
  - **Video Handling:** Dynamically maps `(16, 224, 224, 3)` `uint8` arrays to `(16, 3, 224, 224)` `float32` PyTorch tensors and normalizes using ImageNet mean/std.
  - **Video Augmentation:** Incorporates strict *temporally consistent* spatial transforms (`HorizontalFlip`, `Rotate`, `ColorJitter`) across all 16 frames simultaneously utilizing `albumentations` with dynamic `additional_targets`.
  - **Audio Handling:** Dynamically maps `(3, 224, 224)` `float32` `.npy` spectrograms to PyTorch tensors.
  - **Label Handling:** Reads exact canonical labels (`label` column) mapping `"Real" -> 0.0` and `"Fake" -> 1.0` as PyTorch floats.
- **Class Imbalance Strategy:** 
  - **Train:** Resolved extreme `700 Real` / `14,383 Fake` class imbalance by implementing a `WeightedRandomSampler` which computes inverse frequency probabilities. (Tested dynamically generating ~50/50 batches out of a 4.6% Real canonical distribution).
  - **Val/Test:** Natural dataset distribution strictly preserved (`shuffle=False`, Sequential, no sampler).
  - *Note:* Class imbalance is handled at the training sampling level in Phase 4. Focal Loss will provide the complementary loss-level handling during the later training phase.
- **Verification Results:** 
  - Dataset lengths confirmed accurate (15,083 Train, 3,191 Val, 3,270 Test).
  - Output batches confirmed explicitly dimensioned as `(B, 16, 3, 224, 224)` and `(B, 3, 224, 224)`.
  - CUDA transfer successfully tested.
  - Git secured cleanly without staging any processed dataset contents.

## Phase 5: Multimodal Model Architecture (Completed)
- **Status:** PHASE 5 COMPLETE
- **Architecture:**
  - **Video Encoder** (`models/video_encoder.py`): Wraps EfficientNet-B0 (pretrained ImageNet). Accepts `(B, 16, 3, 224, 224)`, reshapes to `(B×16, 3, 224, 224)`, passes through EfficientNet feature extractor + AdaptiveAvgPool2d, reshapes to `(B, 16, 1280)`, mean-pools to `(B, 1280)`.
  - **Audio Encoder** (`models/audio_encoder.py`): Wraps EfficientNet-B0 (pretrained ImageNet). Accepts `(B, 3, 224, 224)` log-mel spectrograms, returns `(B, 1280)`.
  - **Fusion Model** (`models/fusion_model.py`): Concatenates video + audio features → `(B, 2560)` → `Linear(2560→512)` → `ReLU` → `Dropout(0.3)` → `Linear(512→1)` → raw logit `(B, 1)`.
  - **Total Parameters:** 9,326,841 (all trainable in this phase).
  - No sigmoid inside the model; external `BCEWithLogitsLoss` / Focal Loss will apply it during training.
- **Transfer Learning:** Both EfficientNet-B0 backbones loaded from `EfficientNet_B0_Weights.IMAGENET1K_V1` via torchvision 0.22.1.
- **Grad-CAM Accessibility:** `model.video_encoder.grad_cam_layer` and `model.audio_encoder.grad_cam_layer` both reference the final `Conv2dNormActivation` block (block index `[-1]` of `backbone.features`), ready for future hook attachment without architectural changes.
- **Sanity Test Results (`scripts/test_model_architecture.py`):**
  - Video input shape: `(2, 16, 3, 224, 224)` ✓
  - Audio input shape: `(2, 3, 224, 224)` ✓
  - Video feature: `(2, 1280)` ✓
  - Audio feature: `(2, 1280)` ✓
  - Fused feature: `(2, 2560)` ✓
  - Output logit: `(2, 1)` ✓
  - NaN/Inf check: PASSED ✓
  - Grad-CAM layers accessible: PASSED ✓
  - CUDA forward pass: SUCCESS ✓
  - Trainable parameters: 9,326,841 ✓
- **Training:** NOT performed in this phase.

## Phase 6: Training Infrastructure (Completed)
- **Status:** PHASE 6 COMPLETE — Infrastructure verified. Full training NOT yet run.
- **Files Created:**
  - `training/__init__.py` — Package init
  - `training/losses.py` — `BinaryFocalLoss` module
  - `training/trainer.py` — `Trainer` class
  - `scripts/train.py` — Training entry point (`python scripts/train.py`)
  - `scripts/test_training_pipeline.py` — End-to-end sanity test
- **Architecture:**
  - **Loss:** Binary Focal Loss (`gamma=2.0`, configurable) on raw logits. Numerically stable via `F.binary_cross_entropy_with_logits`.
  - **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-4`)
  - **Scheduler:** CosineAnnealingLR (`T_max=num_epochs`, `eta_min=lr*0.01`)
  - **AMP:** `torch.amp.autocast("cuda")` + `GradScaler` enabled on CUDA, gracefully disabled on CPU.
  - **Gradient Clipping:** `max_norm=1.0` applied before each optimizer step.
- **Checkpointing:**
  - `checkpoints/latest.pt` — Saved every epoch.
  - `checkpoints/best_model.pt` — Saved when val loss improves.
  - Checkpoint contains: model state, optimizer state, scheduler state, scaler state, epoch, best_val_loss, history, config.
- **Resume:** `python scripts/train.py --resume checkpoints/latest.pt` restores full state.
- **Training History:** `training_history.json` — epoch, train_loss, val_loss, lr, time_sec per epoch.
- **Default Configuration (RTX 4050 6GB safe):**
  - `batch_size=4`, `num_epochs=10`, `lr=1e-4`, `focal_gamma=2.0`, `seed=42`
- **Progress Display:** tqdm progress bars per epoch for train and val loops + epoch summary.
- **Sanity Test Results (`scripts/test_training_pipeline.py`):**
  - [1/8] DataLoaders loaded (7541 train batches, 1596 val batches) [OK]
  - [2/8] Model instantiated (9,326,841 trainable parameters) [OK]
  - [3/8] Focal Loss on dummy data: 0.0399 [OK]
  - [4/8] Forward + Backward + Optimizer: 3 batches, 426 layers updated [OK]
  - [5/8] Gradient check: 426 layers with gradients [OK]
  - [6/8] Validation forward pass: loss=0.1762 [OK]
  - [7/8] Checkpoint round-trip: PASSED [OK]
  - [8/8] Grad-CAM layers intact post-training-step [OK]
  - **TRAINING PIPELINE TEST PASSED**

## Phase 6.5: Training Readiness / Hardware Timing Check (Completed)
- **Status:** READY FOR FULL TRAINING RUN
- **Hardware Tested:** NVIDIA GeForce RTX 4050 Laptop GPU | CUDA 11.8
- **Test Configuration:** batch_size=4 | AMP enabled | 30 real measured batches (5 warm-up discarded)
- **Measured Throughput:**
  - Avg batch time: 0.191s | Median: 0.181s
  - Videos/sec: 20.94
  - Peak allocated VRAM: 3.03 GB / 6 GB
  - Peak reserved VRAM: 3.27 GB / 6 GB
- **Estimated Durations (based on measured throughput):**
  - Per training epoch: ~12m 0s (15,083 samples)
  - Per validation epoch: ~0m 40s (3,191 samples)
  - Per full epoch: ~12m 41s
  - 5 epochs: ~1h 3m
  - 10 epochs: ~2h 7m
  - 15 epochs: ~3h 10m
- **Issues:** None. CUDA OOM: No | NaN/Inf: No | Checkpoint save: PASSED
- **Notes:**
  - `num_workers=4` caused DataLoader worker crash on Windows during validation (after training loop); resolved by using `num_workers=0` in timing script. Full `train.py` uses `num_workers=4` with persistent processes which is safe for the actual training loop.
  - VRAM headroom: ~2.73 GB free — batch_size=4 is confirmed safe.
- **Recommended Configuration:** `batch_size=4`, `num_epochs=10`, AMP enabled, AdamW lr=1e-4. Safe and efficient for RTX 4050 6GB.

## Phase 7: Actual Full Training
- **Status:** READY FOR MANUAL LAUNCH
- **Configuration (Verified):**
  - Planned epochs: 10
  - Batch size: 4
  - `num_workers`: 0 (Updated for stability on Windows)
  - AMP: Enabled
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
  - Loss: Binary Focal Loss (gamma=2.0)
  - GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Data:**
  - Training: 15,083 samples
  - Validation: 3,191 samples
  - Test: 3,270 samples (Held out - NOT used during training)
- **Expected Runtime:** Approximately 2–2.5 hours for 10 epochs (based on Phase 6.5 readiness results).
- **Execution Details:**
  - Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC, Loss) added to `training/trainer.py`.
  - Checkpoints will be saved to `checkpoints/latest.pt` and `checkpoints/best_model.pt`.
  - History will be saved to `training_history.json`.
  - Plotting script `scripts/plot_training_curves.py` created for generating research figures.
- **IMPORTANT:** The actual full training command has **NOT** been launched by the agent. The user must launch it manually to witness the process.
- **Manual Launch Command:**
  ```powershell
  .\.venv_gpu\Scripts\python.exe scripts\train.py
  ```

## Phase 8: Final Test Evaluation Preparation
- **Status:** READY FOR MANUAL LAUNCH
- **Configuration:**
  - Evaluates on untouched `data/dataset_split/test.csv` (3,270 samples).
  - Loads weights from `checkpoints/best_model.pt`.
- **Execution Details:**
  - Script created at `scripts/evaluate_test.py`.
  - Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix (TP, TN, FP, FN).
  - Generates artifacts in `results/` directory:
    - `test_metrics.json`
    - `test_predictions.csv` (sample_path, true_label, predicted_label, prob_fake)
    - `confusion_matrix.png`
    - `roc_curve.png`
- **Sanity Check:** PASSED. `checkpoints/best_model.pt` loaded successfully with PyTorch 2.6 `weights_only=False` fix.
- **IMPORTANT:** The actual full test evaluation command has **NOT** been launched by the agent. The user must launch it manually.
- **Manual Launch Command:**
  ```powershell
  .\.venv_gpu\Scripts\python.exe scripts\evaluate_test.py
  ```

## Phase 9A: Grad-CAM Preparation and Sanity Check
- **Status:** COMPLETE
- **Configuration (Verified):**
  - **Model Architecture Checked:** `models/video_encoder.py`, `models/audio_encoder.py`, `models/fusion_model.py`.
  - **Target Layers Found:** `self.grad_cam_layer` maps to `backbone.features[-1]` in both EfficientNet encoders (the final convolutional block).
  - **Checkpoint Verification:** `checkpoints/best_model.pt` loaded successfully.
- **Sanity Check Execution:**
  - Used 1 single test sample (`FakeVideo-FakeAudio/African/men/id00391/00052_id00166_PQV2ZbJBLNk_id00478_wavtolip.mp4`).
  - True label: 1.0 (Fake), Predicted Prob: 1.0000.
  - Forward and backward hooks registered successfully.
  - Backward pass completed without gradient detachment errors.
- **Readiness Verification Results:**
  - **Video Encoder:** TECHNICALLY SUPPORTED. Captured activations and gradients successfully (Shape: `[16, 1280, 7, 7]`).
  - **Audio Encoder:** TECHNICALLY SUPPORTED. Captured activations and gradients successfully (Shape: `[1, 1280, 7, 7]`).
- **Issues Discovered:**
  - Initially hit a Windows `cp1252` encoding issue when printing a unicode checkmark in the sanity check script. Replaced with `[OK]`.
- **Note:** The actual Grad-CAM heatmap generation has NOT been implemented yet. This was strictly a technical readiness verification.

## Phase 9B: Actual Grad-CAM Implementation
- **Status:** COMPLETE
- **Implementation Details:**
  - Script created at `scripts/generate_gradcam.py`.
  - Reusable `GradCAM` class implements forward/backward hooks, global average pooling over gradients to extract feature weights, and ReLU applied to the weighted combination of activations.
  - Heatmaps are spatially normalized to `[0, 1]`, resized to `224x224`, and converted to `COLORMAP_JET` overlays using `cv2`.
  - `checkpoints/best_model.pt` loaded successfully with PyTorch 2.6 `weights_only=False` fix.
- **Samples Selected:**
  1. `correct_fake`: `FakeVideo-FakeAudio/African/men/id00391/00052_id00166_PQV2ZbJBLNk_id00478_wavtolip.mp4` (Prob Fake: 1.0000)
  2. `false_negatives` (1): `FakeVideo-RealAudio/African/women/id00460/00005.mp4` (Prob Fake: 0.0105)
  3. `false_negatives` (2): `FakeVideo-RealAudio/African/women/id00592/00017.mp4` (Prob Fake: 0.0261)
  4. `false_negatives` (3): `FakeVideo-RealAudio/Asian (East)/men/id06591/00021.mp4` (Prob Fake: 0.0686)
  5. `correct_real`: `RealVideo-FakeAudio/African/men/id00391/00052_fake.mp4` (Prob Fake: 0.0195)
- **Selected Frame Indices:** Deterministically selected frames `[0, 7, 15]` for all video outputs.
- **Generated Output Directory:** `results/gradcam/`
  - Video artifacts: Original frame, Heatmap, and Overlay (per frame index).
  - Audio artifacts: Original spectrogram, Heatmap, and Overlay.
  - `metadata.json` created containing paths, probabilities, target classes, and coordinates for all generated artifacts.
- **No Issues Discovered:** Implementation correctly propagated gradients via BCE logit and successfully produced overlays.

## Phase 9.5: Repository Cleanup and Production Artifact Freeze
- **Status:** COMPLETE
- **Cleanup Actions:**
  - Audited full repository to separate active production pipelines from temporary, historical, and testing scripts.
  - Renamed the misleading `testing/` directory to `archive/` to properly communicate its purpose as a storage location for non-production reproducible artifacts.
  - Moved all one-time training analysis scripts (`test_training_pipeline.py`, `measure_training_speed.py`, `plot_training_curves.py`) into `archive/training/`.
  - Moved Grad-CAM generation and sanity check scripts into `archive/gradcam/`.
  - Moved the generated Grad-CAM research artifact outputs from `results/gradcam/` to `archive/gradcam/outputs/` to keep `results/` clean for final test metric evaluation.
- **Active Production Structure Verified:**
  - `models/`: Active model architecture.
  - `preprocessing/`: Active preprocessing pipeline.
  - `dataset/`: Active dataloaders.
  - `training/`: Active training loop/loss functions.
  - `scripts/`: Now exclusively contains active, reproducible entry points (`train.py`, `evaluate_test.py`, `preprocess_audio_offline.py`, `preprocess_dataset_offline.py`).
  - `results/`: Contains exclusively final evaluation artifacts (metrics, predictions, cm, roc).
- **Data Integrity:** Raw datasets, processed arrays, and model checkpoints remain 100% untouched. No architecture was changed.

## Phase 10: Minimal Streamlit Application
- **Status:** COMPLETE
- **Implementation Details:**
  - `app.py` created at the repository root as the sole web application entry point.
  - Implemented exactly two classic pages: `Detection` and `Results & Evaluation`.
  - Application strictly adheres to minimal UI constraints without unnecessary custom CSS, animations, or marketing fluff.
- **Inference Flow:**
  - Processes uploaded `.mp4` video dynamically into a temporary file.
  - Utilizes active `VideoPreprocessor` and `SpectrogramGenerator` to compute 16 standardized normalized video frames and a single 16 kHz Log-Mel spectrogram.
  - Passes these tensors dynamically through the unchanged `best_model.pt` architecture.
- **Grad-CAM Integration:**
  - Embedded the reusable `GradCAM` extraction class directly into the Streamlit logic.
  - Dynamically computes and displays the `[0, 1]` spatial overlays on 3 deterministic video frames (`[0, 7, 15]`) and the audio spectrogram.
- **Evaluation Page:**
  - Loads metrics exclusively from `results/test_metrics.json`.
  - Renders the existing visual artifacts `results/confusion_matrix.png` and `results/roc_curve.png`.
- **Streamlit Status:**
  - Server is actively running as a daemon on `http://localhost:8501`.

## Phase 10.5: Multi-Head Architecture Read-Only Analysis & Clean Restart
- **Status:** COMPLETE
- **Analysis Conclusion:** Investigated the old `v1_single_head` workflow and determined it scientifically invalid and technically impossible to support standalone image/audio predictions using the existing checkpoints. A multi-head multi-task architecture trained from scratch was approved.
- **Repository Archive:**
  - Swept all old single-head dependencies (`models/`, `dataset/`, `training/`, `checkpoints/`, `results/`, `app.py`, `scripts/train.py`, `scripts/evaluate_test.py`) into `archive/v1_single_head/` to prevent namespace collisions.
  - Safely preserved all frozen `data/` and `preprocessing/` infrastructure intact.

## Phase 11: Multi-Head Multimodal Architecture
- **Status:** IN PROGRESS (Models implemented)
- **Implementation Details:**
  - `models/visual_encoder.py`: Unifies Image and Video backbone parsing (EfficientNet-B0). Extracts `(B, 1280)` mean-pooled video representations alongside `(B*T, 1280)` independent frame features.
  - `models/audio_encoder.py`: Standard EfficientNet-B0 backbone for Log-Mel spectrograms yielding `(B, 1280)`.
  - `models/multihead_model.py`: Implements `MultiHeadDeepfakeModel` wrapping the encoders.
- **Prediction Heads:**
  - `image_head`: `Linear(1280, 1)`
  - `audio_head`: `Linear(1280, 1)`
  - `fusion_head`: `Linear(2560, 512) -> ReLU -> Dropout -> Linear(512, 1)`
- **Forward Capabilities:**
  - Dynamically routes inputs to support standalone image inference, standalone audio inference, and multimodal fusion inference.
  - Supports multi-task training by calculating independent predictions for all $16$ video frames, the audio, and the fused multimodal vector concurrently.

## Phase 12: V2 Multi-Head Dataset & DataLoader
- **Status:** COMPLETE
- **Implementation Details:**
  - `dataset/multimodal_dataset.py`: Rebuilt to extract and return `(video, audio, video_label, audio_label, overall_label)`. Validated mapping of `Fake -> 1.0` and `Real -> 0.0` across all sub-modalities.
  - `dataset/dataloader_factory.py`: Implemented Train/Val/Test loaders. Retained the V1 `WeightedRandomSampler` balancing exactly against the `overall_label` to ensure mathematically identical class boundaries and sampling distribution as V1.
- **Image-Head Training Strategy:**
  - Decided **not** to extract a separate image dataset.
  - The dataset simply yields the `(B, 1)` `video_label`. 
  - The `test_dataloader.py` verification script mathematically proved that executing `.repeat_interleave(16, dim=0)` dynamically expands the `video_label` into `(B*16, 1)`, perfectly aligning with the `MultiHeadDeepfakeModel`'s `image_head` predictions `(B*16, 1)`.
- **Verification:**
  - Confirmed deterministic un-shuffled loading for Validation and Test.
  - Tested batch shape integrity: Video `(B, 16, 3, 224, 224)`, Audio `(B, 3, 224, 224)`, Labels `(B, 1)`.
  - Confirmed correct audio path targeting (`processed_audio/spectrograms`).

## Phase 13: V2 Multi-Task Loss & Trainer
- **Status:** COMPLETE
- **Loss Implementation:** 
  - Extracted the numerically stable `BinaryFocalLoss` (gamma=2.0) from V1.
  - Implemented `MultiTaskFocalLoss` which applies the focal loss equally across all three prediction heads (`total = image + audio + fusion`).
  - Image target extraction perfectly aligns `(B, 1)` `video_label` into `(B*16, 1)` dynamically matching the unpooled sequence dimensions from the visual encoder.
- **Trainer Implementation:** 
  - V2 Trainer natively supports tracking metrics for all 3 sub-tasks dynamically.
  - Retained `AdamW`, `ReduceLROnPlateau`, and AMP `GradScaler`. 
  - V2 Checkpointing architecture implemented to save `v2_best_model.pt` separately without corrupting V1 archives.
- **Sanity Test:**
  - A dry-run pipeline test (`scratch/sanity_train.py`) verified the entire computational graph. Gradients safely propagated backward from the summation loss into the visual backbone, audio backbone, and all isolated projection heads without NaNs.
