# Forward-Fill Quality Check Report

## 1. Methodology
To validate the quality of the bounding boxes produced by the forward-filling mechanism, we ran a specialized diagnostic script (`scripts/quality_check_forward_fill.py`) on the same 50-video / 800-frame deterministic sample. The script quantified consecutive forward-fill streaks and measured the "drift" (geometric distance) when a forward-fill streak is broken by a new direct detection. It also generated full 16-frame annotated visual sequences for five representative cases for manual inspection.

## 2. Representative Visual Findings
We examined full sequences for the following representative cases:
- **Mixed Direct and FF (`vid3`)**: Frequent toggling between direct and forward-filled.
- **Mostly FF (`vid17`)**: High proportion of forward-filled frames.
- **Longest FF Streak (`vid25`)**: Contains the maximum measured streak of 12 consecutive forward-filled frames.
- **High Center Fallback (`vid13`)**: Exhibits a high center fallback rate despite having some detections.
- **0 Detections (`vid7`)**: 100% center fallback.

**Observations:**
- **Face Containment**: Forward-filled bounding boxes consistently remain on the face due to the conservative 20% margin, accommodating minor head movements.
- **Drift**: In most sequences, the face stays relatively static (mean drift of 1.63%). However, there is at least one instance of severe movement (max drift of 34.84%).
- **Clipping and Margins**: The 20% margin is sufficient to prevent the face from leaving the crop boundary during short streaks.

## 3. Quantitative Diagnostics
- **Max Consecutive FF Streak**: 12 frames
- **Mean Consecutive FF Streak**: ~2.99 frames
- **Max Bounding-Box Drift**: 34.84% (distance relative to image diagonal)
- **Mean Bounding-Box Drift**: 1.63%

## 4. Remaining Center-Fallback Analysis
The 33.1% remaining center-fallback frames are overwhelmingly concentrated in:
1. **Zero-Detection Videos**: 8 out of 50 videos (like `vid7`) have 0 direct detections across all 16 frames, making forward-filling impossible.
2. **Late Initial Detections**: Videos (like `vid13`) where the first direct detection occurs late in the sequence (e.g., Frame 5 or 6). Any frames before the first detection cannot be forward-filled and must fall back to a center crop.

## 5. Answers to Key Practical Questions
**A. Do forward-filled crops generally remain on the face?**
Yes. The mean geometric drift between a stale forward-filled box and the next direct detection is only 1.63%, well within the 20% margin buffer.

**B. Are there obvious cases of severe bounding-box drift?**
Yes, but they are rare. We measured a maximum drift of 34.8% in one instance, indicating significant subject movement during a prolonged detection failure.

**C. How long are the longest forward-fill runs?**
The maximum run was 12 consecutive frames (out of 16). The average run is roughly 3 frames.

**D. What is causing the remaining 33.1% center fallback?**
The inability to "backward-fill". If the first frame(s) of a video fail detection, there is no prior bounding box to reuse, forcing a center fallback until the first successful detection occurs. Videos with zero detections (16% of the sample) also contribute heavily.

**E. Is the current video preprocessing pipeline good enough to run on all 21,544 videos?**
No, not quite yet. A 33.1% blind center fallback rate is still too high, especially since it is concentrated at the critical beginning of videos.

**F. If not, identify the ONE most important remaining issue to address.**
**Backward-filling.** We must allow initial frames that fail detection to borrow the bounding box from the *first successful detection* later in the video, rather than defaulting to a center crop.

## 6. Recommendation
Implement a two-pass fallback approach:
1. First pass: Record all direct detections.
2. Second pass: Forward-fill any missing frames after a detection, and **backward-fill** any missing frames before the first detection.
This should nearly eliminate center fallbacks for any video that has at least *one* valid detection.
