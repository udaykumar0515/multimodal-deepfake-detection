"""Validate the frozen FakeAVCeleb canonical dataset manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

SPLITS = ("train", "val", "test")
EXPECTED_TYPES = {
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
}
CANONICAL_COLUMNS = {
    "source", "target1", "target2", "method", "category", "type", "race",
    "gender", "path", "Unnamed: 9", "video_label", "audio_label", "label",
    "sample_path", "split", "metadata_row_count", "method_annotations",
    "metadata_provenance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("Dataset_FakeAVCeleb/meta_data.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset_FakeAVCeleb"))
    parser.add_argument("--original-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/splits/canonical"))
    parser.add_argument("--media-report", type=Path, default=Path("reports/media_integrity_failures.csv"))
    return parser.parse_args()


def load_manifests(directory: Path, canonical: bool) -> dict[str, pd.DataFrame]:
    result = {}
    for split in SPLITS:
        path = directory / f"{split}.csv"
        dataframe = pd.read_csv(path)
        required = set(CANONICAL_COLUMNS)
        if canonical:
            required.discard("method")
        else:
            required -= {"metadata_row_count", "method_annotations", "metadata_provenance"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        result[split] = dataframe
    return result


def check_labels(dataframe: pd.DataFrame) -> list[str]:
    failures = []
    expected_video = dataframe["type"].str.startswith("FakeVideo-").map({True: "Fake", False: "Real"})
    expected_audio = dataframe["type"].str.endswith("-FakeAudio").map({True: "Fake", False: "Real"})
    for column, expected in (("video_label", expected_video), ("audio_label", expected_audio), ("label", expected_video)):
        if not dataframe[column].astype(str).equals(expected.astype(str)):
            failures.append(f"{column} does not match type-derived label")
    if not bool(dataframe["type"].isin(EXPECTED_TYPES).all()):
        failures.append("unexpected type value")
    if dataframe[["video_label", "audio_label", "label"]].isna().any().any():
        failures.append("missing label")
    return failures


def media_check(canonical: dict[str, pd.DataFrame], dataset_root: Path, report_path: Path) -> tuple[int, list[dict[str, str]]]:
    failures = []
    total = 0
    for split in SPLITS:
        for sample_path in canonical[split]["sample_path"]:
            total += 1
            physical_path = dataset_root / sample_path
            capture = cv2.VideoCapture(str(physical_path))
            opened = capture.isOpened()
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
            duration = frame_count / fps if fps > 0 else 0.0
            frame_ok, _ = capture.read() if opened else (False, None)
            capture.release()
            if not opened or frame_count <= 0 or fps <= 0 or duration <= 0 or not frame_ok:
                failures.append({
                    "split": split,
                    "sample_path": sample_path,
                    "opened": str(opened),
                    "frame_count": str(frame_count),
                    "fps": str(fps),
                    "duration_seconds": str(duration),
                    "first_frame_read": str(frame_ok),
                })
        print(f"Media integrity checked: {split} ({len(canonical[split])} videos)")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["split", "sample_path", "opened", "frame_count", "fps", "duration_seconds", "first_frame_read"]).to_csv(report_path, index=False)
    return total, failures


def main() -> int:
    args = parse_args()
    original = load_manifests(args.original_dir, canonical=False)
    canonical = load_manifests(args.canonical_dir, canonical=True)
    original_all = pd.concat(original.values(), ignore_index=True)
    canonical_all = pd.concat(canonical.values(), ignore_index=True)
    failures: list[str] = []
    if len(original_all) != 21566:
        failures.append("unexpected original row count")
    if len(original_all) != int(canonical_all["metadata_row_count"].sum()):
        failures.append("canonical provenance row counts do not account for original rows")
    if original_all["sample_path"].nunique() != len(canonical_all):
        failures.append("canonical row count does not equal unique original paths")
    if canonical_all["sample_path"].duplicated().any():
        failures.append("duplicate canonical sample path")
    split_paths = {split: set(canonical[split]["sample_path"]) for split in SPLITS}
    split_ids = {split: set(canonical[split]["source"]) for split in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if split_paths[left] & split_paths[right]:
                failures.append(f"canonical path overlap: {left}/{right}")
            if split_ids[left] & split_ids[right]:
                failures.append(f"canonical identity overlap: {left}/{right}")
    if len(set().union(*split_ids.values())) != original_all["source"].nunique():
        failures.append("canonical identity coverage mismatch")
    if canonical_all["source"].isna().any() or canonical_all["source"].eq("").any():
        failures.append("missing canonical source identity")
    for provenance in canonical_all["metadata_provenance"]:
        try:
            records = json.loads(provenance)
        except (TypeError, json.JSONDecodeError):
            failures.append("invalid metadata provenance JSON")
            break
        if not isinstance(records, list) or not records:
            failures.append("empty metadata provenance")
            break
    for split in SPLITS:
        if (canonical[split]["split"] != split).any():
            failures.append(f"incorrect split value in canonical {split}")
        failures.extend(f"{split}: {failure}" for failure in check_labels(canonical[split]))
    invalid_paths = canonical_all["sample_path"].apply(lambda value: Path(value).is_absolute() or Path(value).suffix.lower() != ".mp4")
    if invalid_paths.any():
        failures.append(f"invalid canonical paths: {int(invalid_paths.sum())}")
    missing_paths = [path for path in canonical_all["sample_path"] if not (args.dataset_root / path).is_file()]
    if missing_paths:
        failures.append(f"missing physical files: {len(missing_paths)}")
    media_total, media_failures = media_check(canonical, args.dataset_root, args.media_report)
    if media_failures:
        failures.append(f"media integrity failures: {len(media_failures)}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "original_metadata_rows": len(original_all),
        "unique_physical_videos": int(original_all["sample_path"].nunique()),
        "canonical_videos": len(canonical_all),
        "canonical_by_split": {split: len(canonical[split]) for split in SPLITS},
        "original_rows_by_split": {split: len(original[split]) for split in SPLITS},
        "identity_counts": {split: int(canonical[split]["source"].nunique()) for split in SPLITS},
        "media_checked": media_total,
        "media_failures": len(media_failures),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
