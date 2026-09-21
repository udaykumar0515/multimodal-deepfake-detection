# Video Preprocessing Quality Validation Report

## 1. Validation Methodology
A comprehensive quality validation of the video preprocessing pipeline was conducted to evaluate the reliability of RetinaFace and the central-crop fallback mechanism before committing to full-dataset processing (21,544 videos).

**Sample Selection:**
- Total Videos: 50
- Distribution: 25 Train, 15 Validation, 10 Test
- The selection was done using a deterministic fixed seed (`np.random.seed(42)`) to ensure reproducibility.
- The sample explicitly included a mixture of "Real" and "Fake" labels across the different dataset splits.

**Configuration:**
- Temporal Sampling: `UniformTemporalSampler` (16 frames per video, spanning the entire duration).
- Face Detection: `RetinaFaceCropper` (using `insightface` det_10g).
- Margin: 20% expansion of the bounding box.
- Target Size: 224x224.
- Fallback Mechanism: Static central crop (clamped to image boundaries) when detection fails.

## 2. Hardware and Environment
- **CUDA Available:** False (Running entirely on CPU)
- **ONNX Runtime Providers:** `['AzureExecutionProvider', 'CPUExecutionProvider']`
- **Implication:** InsightFace (RetinaFace) is executing purely on the CPU, which heavily impacts its internal confidence thresholds and speed, potentially leading to missed detections on blurry or fast-moving frames.

## 3. Quantitative Statistics
**Aggregate Totals:**
- **Total Videos Tested:** 50
- **Total Frames Processed:** 800 (50 videos × 16 frames)
- **Total Successful Detections:** 332
- **Total Fallback Frames:** 468

**Rates:**
- **Detection Success Rate:** 41.5%
- **Fallback Rate:** 58.5%

**Problematic Video Thresholds:**
- **Videos with 0 successful detections (100% fallback):** 8
- **Videos with fewer than 8 successful detections (<50%):** 31
- **Videos with excessive fallback (>50% frames):** 31

**Per-Video Metrics:**
- **Detections:** Min: 0 | Max: 16 | Mean: 6.64
- **Fallbacks:** Min: 0 | Max: 16 | Mean: 9.36

**Timing (CPU):**
- **Average Processing Time per Video:** ~4.57 seconds
- **Average Processing Time per Frame:** ~0.28 seconds
- **Total Validation Time:** ~3 minutes 48 seconds

## 4. Visual Inspection and Fallback Assessment
Visual debug images (original frame, detected bounding boxes, and final 224x224 crop) were generated and saved for the first 5 videos in `reports/video_preprocessing_validation/visual_samples/`.

**Observations on Fallback:**
The current fallback is a static center crop of 224x224 pixels. While it guarantees that the data loading pipeline does not crash (preventing NaNs and dimensional errors), a 58.5% fallback rate is structurally unacceptable for training a deepfake detection model. If more than half the frames in a video are just arbitrary center crops, the model will learn background noise instead of facial blending artifacts.

## 5. Conclusions and Recommendations
The current pipeline architecture is technically sound (it successfully outputs `[16, 3, 224, 224]` tensors without crashing), but the **quality of the data being extracted is severely compromised by a 58.5% RetinaFace failure rate**.

**Root Cause:**
RetinaFace via `insightface` on `CPUExecutionProvider` is likely struggling with motion blur, side-profiles, and lighting conditions without GPU-accelerated tensor precision.

**Recommendations before Full-Dataset Processing:**
1. **DO NOT proceed to process all 21,544 videos with this exact setup.**
2. **Attempt GPU Acceleration:** We must resolve the CUDA unavailability for `onnxruntime-gpu` / PyTorch so that InsightFace can use `CUDAExecutionProvider`. This often drastically improves both speed and detection accuracy/confidence.
3. **Alternative Fallback / Forward-Filling:** If a face is detected in Frame 3 but missed in Frame 4, the fallback should ideally use the bounding box coordinates from Frame 3 (forward-filling) rather than defaulting to a blind center crop. This would salvage many of the 468 fallback frames.

## 6. Phase 1.7: Bounding-Box Forward-Fill Experiment
To address the high fallback rate (58.5%), we implemented a stateful "Bounding-Box Forward-Filling" mechanism. When RetinaFace fails to detect a face in frame $N$, the system now reuses the bounding box from the most recent successful detection in the sequence ($N-k$) and applies the same 20% margin to crop the frame.

### Results
- **Direct Detections:** 332 frames (41.5%)
- **Forward-Filled Frames:** 203 frames (25.4%)
- **Center Fallbacks (Blind):** 265 frames (33.1%)

### Impact
- **Fallback Rate Reduction:** The blind center-crop fallback rate was successfully reduced from **58.5%** down to **33.1%**.
- **Fully Covered Videos:** 20 out of 50 videos (40%) now have zero center-fallback frames (i.e., every frame was either directly detected or forward-filled).
- **Severely Impacted Videos:** The number of videos with >50% center-fallback frames dropped from 31 to 15.

### Conclusion
Forward-filling successfully salvaged nearly half of the failed detections (203 out of 468). This is a massive improvement in data quality without increasing computational overhead. We are now in a much better position to proceed with full-dataset processing.

## 7. Phase 1.8: Forward-Fill Quality Check
To ensure the 25.4% forward-filled frames were producing valid face crops, we ran a targeted visual diagnostic (`scripts/quality_check_forward_fill.py`) on the 50-video sample.

### Key Findings
- **Face Containment**: Forward-filled crops generally remain accurately centered on the face. The 20% margin is sufficient.
- **Drift**: The mean geometric drift between a stale forward-filled box and the next direct detection is only **1.63%**, confirming that the face is mostly static during detection failures. However, rare instances of severe drift (up to 34.8%) were observed.
- **Run Length**: The maximum run of consecutive forward-filled frames was 12 (out of 16). The mean run was ~3 frames.
- **Remaining Center Fallbacks (33.1%)**: The primary cause of the remaining center fallbacks is the inability to "backward-fill". Videos with no detections (8/50) or videos where the initial frames fail detection cannot be forward-filled, forcing a center crop.

### Recommendation
Before processing all 21,544 videos, we must address the remaining 33.1% fallback rate by implementing **Backward-Filling**. This will allow initial frames that fail detection to borrow the bounding box from the *first successful detection* later in the sequence.
