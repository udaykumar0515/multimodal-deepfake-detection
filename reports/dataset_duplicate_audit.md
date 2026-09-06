# Dataset Duplicate Metadata Audit

- **Audit date:** 2026-09-06
- **Source metadata:** `Dataset_FakeAVCeleb/meta_data.csv` (represented by the current split manifests)
- **Split manifests:** `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`

## Summary

- Total manifest rows: **21566**
- Unique canonical sample paths: **21544**
- Duplicated canonical paths: **22**
- Duplicate rows: **44**
- Rows by split: train=15099, val=3194, test=3273
- Identities by split: train=350, val=75, test=75

## Duplicate Findings

The table includes all relevant metadata values for each duplicated canonical path. A semicolon-separated value means the two rows differ for that field.

| sample_path | source | split | occurrences | physical file | exact metadata | differing fields | type | method | video label | audio label | label | category | race | gender | path | target1 | target2 | metadata directory |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FakeVideo-FakeAudio/African/women/id00220/00027_id03658_wavtolip.mp4 | id00220 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00027_id03658_wavtolip.mp4 | id03658 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id00220 |
| FakeVideo-FakeAudio/African/women/id00460/00005_id00460_wavtolip.mp4 | id00460 | test | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00005_id00460_wavtolip.mp4 | id00460 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id00460 |
| FakeVideo-FakeAudio/African/women/id00592/00017_id01661_wavtolip.mp4 | id00592 | test | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00017_id01661_wavtolip.mp4 | id01661 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id00592 |
| FakeVideo-FakeAudio/African/women/id01532/00065_id02824_wavtolip.mp4 | id01532 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00065_id02824_wavtolip.mp4 | id02824 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id01532 |
| FakeVideo-FakeAudio/African/women/id01661/00059_id00577_wavtolip.mp4 | id01661 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00059_id00577_wavtolip.mp4 | id00577 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id01661 |
| FakeVideo-FakeAudio/African/women/id01661/00059_id04055_wavtolip.mp4 | id01661 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00059_id04055_wavtolip.mp4 | id04055 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id01661 |
| FakeVideo-FakeAudio/African/women/id01783/00015_id03569_wavtolip.mp4 | id01783 | val | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00015_id03569_wavtolip.mp4 | id03569 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id01783 |
| FakeVideo-FakeAudio/African/women/id02617/00028_id00592_wavtolip.mp4 | id02617 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | African | women | 00028_id00592_wavtolip.mp4 | id00592 | - | FakeAVCeleb/FakeVideo-FakeAudio/African/women/id02617 |
| FakeVideo-FakeAudio/Asian (East)/men/id00863/00069_id02561_wavtolip.mp4 | id00863 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00069_id02561_wavtolip.mp4 | id02561 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id00863 |
| FakeVideo-FakeAudio/Asian (East)/men/id00863/00069_id06269_wavtolip.mp4 | id00863 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00069_id06269_wavtolip.mp4 | id06269 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id00863 |
| FakeVideo-FakeAudio/Asian (East)/men/id05332/00065_id04774_wavtolip.mp4 | id05332 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00065_id04774_wavtolip.mp4 | id04774 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id05332 |
| FakeVideo-FakeAudio/Asian (East)/men/id05383/00015_id05743_wavtolip.mp4 | id05383 | val | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00015_id05743_wavtolip.mp4 | id05743 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id05383 |
| FakeVideo-FakeAudio/Asian (East)/men/id05479/05479_id02561_wavtolip.mp4 | id05479 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 05479_id02561_wavtolip.mp4 | id02561 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id05479 |
| FakeVideo-FakeAudio/Asian (East)/men/id05479/05479_id08613_wavtolip.mp4 | id05479 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 05479_id08613_wavtolip.mp4 | id08613 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id05479 |
| FakeVideo-FakeAudio/Asian (East)/men/id06591/00021_id06591_wavtolip.mp4 | id06591 | test | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00021_id06591_wavtolip.mp4 | id06591 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id06591 |
| FakeVideo-FakeAudio/Asian (East)/men/id06594/00002_id00056_wavtolip.mp4 | id06594 | val | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00002_id00056_wavtolip.mp4 | id00056 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id06594 |
| FakeVideo-FakeAudio/Asian (East)/men/id06776/00021_id04789_wavtolip.mp4 | id06776 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00021_id04789_wavtolip.mp4 | id04789 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id06776 |
| FakeVideo-FakeAudio/Asian (East)/men/id06878/00001_id06776_wavtolip.mp4 | id06878 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00001_id06776_wavtolip.mp4 | id06776 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id06878 |
| FakeVideo-FakeAudio/Asian (East)/men/id08613/00074_id03965_wavtolip.mp4 | id08613 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00074_id03965_wavtolip.mp4 | id03965 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id08613 |
| FakeVideo-FakeAudio/Asian (East)/men/id08613/00074_id06470_wavtolip.mp4 | id08613 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00074_id06470_wavtolip.mp4 | id06470 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id08613 |
| FakeVideo-FakeAudio/Asian (East)/men/id08652/00006_id00056_wavtolip.mp4 | id08652 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00006_id00056_wavtolip.mp4 | id00056 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id08652 |
| FakeVideo-FakeAudio/Asian (East)/men/id08652/00006_id02561_wavtolip.mp4 | id08652 | train | 2 | yes | no | method | FakeVideo-FakeAudio | faceswap-wav2lip; wav2lip | Fake | Fake | Fake | D | Asian (East) | men | 00006_id02561_wavtolip.mp4 | id02561 | - | FakeAVCeleb/FakeVideo-FakeAudio/Asian (East)/men/id08652 |

## Verification

- All split CSVs loaded with the expected schema: **yes**.
- Every duplicated path resolves to an existing physical file: **yes** (0 missing).
- Cross-split duplicate sample paths: **0**.
- Cross-split source identities: **0**.
- Every manifest `split` value matches its file: **yes**.
- Duplicate groups with metadata differences: **22**.
- Duplicate groups with visual/audio/final label conflicts: **0**.

## Decision

- Classification: **same physical video with conflicting metadata annotations**, not exact duplicate metadata records. Every conflict is in `method`: `faceswap-wav2lip` versus `wav2lip`.
- Label result: all duplicate groups agree on `video_label`, `audio_label`, and `label`; no label correction was performed.
- Recommended action: **retain the current manifests unchanged pending dataset review**. Removing one row would reduce the row-level weighting of that physical file and discard one method annotation; retaining both preserves the source metadata but can cause the same physical video to be sampled twice during later preprocessing.
- Automatic deduplication: **not safe** because metadata conflicts exist. No split CSV or raw dataset file was modified by this audit.

**Final status:** PASS - audit complete; manifests unchanged
