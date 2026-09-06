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