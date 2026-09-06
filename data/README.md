# Dataset Manifests

The raw FakeAVCeleb dataset is expected to exist locally under `Dataset_FakeAVCeleb/`. Raw videos are intentionally excluded from GitHub.

## Manifest layers

- `splits/train.csv`, `splits/val.csv`, and `splits/test.csv` are the original identity-based manifests. They preserve every metadata row from `meta_data.csv`, including the 22 physical paths represented by two method annotations.
- `splits/canonical/train.csv`, `splits/canonical/val.csv`, and `splits/canonical/test.csv` are the frozen manifests for future preprocessing. They contain exactly one row per physical video path.

Canonicalization collapses duplicate physical paths without discarding provenance. The combined `method_annotations` field preserves all methods for a path, and `metadata_provenance` stores the original metadata records as deterministic JSON. `metadata_row_count` records how many original rows contributed to the canonical sample.

The visual label is `label` and follows the video component of `type`: `RealVideo-*` is `Real`, and `FakeVideo-*` is `Fake`. `audio_label` retains the audio component separately.

Future preprocessing must consume the canonical manifests and resolve each `sample_path` relative to the local `Dataset_FakeAVCeleb/` directory. No train, validation, or test video directories should be created.

## Regeneration and validation

From the repository root:

```text
python scripts/create_identity_split.py
python scripts/audit_duplicate_records.py
python scripts/create_canonical_manifests.py
python scripts/validate_dataset.py
python scripts/create_final_dataset_summary.py
```

The validation command checks manifest accounting, identity isolation, labels, relative `.mp4` paths, physical-file existence, and lightweight OpenCV container/readability. It does not extract frames or audio.
