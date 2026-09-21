# Phase 2 Verification Instructions & Log

## Full Run Results (Reported by User)
```text
- Expected videos: 21,544
- Successful: 21,544
- Failed: 0
- Runtime: ~9h 12m
- Expected frames: 21,544 × 16 = 344,704
- Reported frame statistics:
  Direct: 141,166
  Forward-fill: 83,946
  Center fallback: 119,592
- ONNX Runtime produced CUDA provider errors:
  LoadLibrary failed with error 126 while loading onnxruntime_providers_cuda.dll
```

## Specific Verification Instructions Provided
The user requested ONE COMPLETE POST-PROCESSING VERIFICATION covering all the following sections without modifying any files or doing partial checks:

1. **DATASET COVERAGE**: Verify 21,544 processed outputs exist and match CSVs exactly.
2. **OUTPUT FORMAT**: Verify the exact format (`.npy` vs `.jpg`), shape, dtype, NaN/Infs, and disk usage.
3. **FRAME COUNT**: Verify exactly 344,704 frames exist across the outputs.
4. **VIDEO-TO-OUTPUT MAPPING**: Verify deterministic naming and no collisions.
5. **LABEL CONSISTENCY**: Verify visual labels from manifests map correctly.
6. **SPLIT ISOLATION**: Verify train/val/test are completely disjoint.
7. **PREPROCESSING METHODOLOGY**: Confirm 16 frames, RetinaFace det_10g, 20% margin, direct/FF/fallback, 224x224, raw dataset untouched.
8. **FORWARD-FILL / FALLBACK ACCOUNTING**: Verify 141,166 + 83,946 + 119,592 sum to 344,704 exactly, and code logic supports it.
9. **CUDA / GPU ISSUE**: Determine definitively if GPU was used despite the Error 126.
10. **RAW DATA INTEGRITY**: Check filesystem timestamps to ensure FakeAVCeleb was untouched.
11. **DISK USAGE**: Calculate total size and average size per video/frame.
12. **FAILURE LOG**: Inspect `failures.json`.
13. **RESUMABILITY**: Confirm skip logic matches `.npy` output.
14. **DOWNSTREAM TRAINING COMPATIBILITY**: Verify compatibility with future PyTorch Dataset (shape, dtype, normalizations).
15. **VISUAL SANITY CHECK**: Check a small representative sample to ensure images are not corrupt/black.
16. **PROJECT STRUCTURE / GIT**: Confirm outputs are untracked.
17. **PROJECT_LOG**: Check if Phase 2 completion is logged.

**Response Format Mandated:**
A single consolidated report structured with sections A through K.
