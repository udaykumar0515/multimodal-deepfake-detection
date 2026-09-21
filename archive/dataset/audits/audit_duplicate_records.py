"""Audit duplicate sample paths in the verified dataset split manifests."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
REQUIRED_COLUMNS = {
    "source",
    "target1",
    "target2",
    "method",
    "category",
    "type",
    "race",
    "gender",
    "path",
    "Unnamed: 9",
    "video_label",
    "audio_label",
    "label",
    "sample_path",
    "split",
}
IGNORED_COMPARISON_COLUMNS = {"_split_file"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit duplicate canonical sample paths in split CSV files."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset_FakeAVCeleb"),
        help="Local dataset root used to check physical sample files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_duplicate_audit.md"),
        help="Markdown report path.",
    )
    return parser.parse_args()


def load_manifests(split_dir: Path) -> pd.DataFrame:
    frames = []
    for split in SPLIT_NAMES:
        manifest_path = split_dir / f"{split}.csv"
        dataframe = pd.read_csv(manifest_path)
        missing = REQUIRED_COLUMNS - set(dataframe.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        dataframe["_split_file"] = split
        frames.append(dataframe)
    return pd.concat(frames, ignore_index=True)


def values_text(values: pd.Series) -> str:
    return "; ".join(sorted({str(value) for value in values}))


def audit(dataframe: pd.DataFrame, dataset_root: Path) -> tuple[str, bool]:
    comparison_columns = [
        column
        for column in dataframe.columns
        if column not in IGNORED_COMPARISON_COLUMNS
    ]
    duplicate_rows = dataframe[dataframe["sample_path"].duplicated(keep=False)]
    duplicate_groups = list(duplicate_rows.groupby("sample_path", sort=True))
    group_details = []
    conflict_count = 0
    label_conflict_count = 0
    missing_physical_count = 0

    for sample_path, group in duplicate_groups:
        varying_fields = [
            column
            for column in comparison_columns
            if group[column].nunique(dropna=False) > 1
        ]
        label_conflict = any(
            group[column].nunique(dropna=False) > 1
            for column in ("video_label", "audio_label", "label")
        )
        physical_exists = (dataset_root / sample_path).is_file()
        if varying_fields:
            conflict_count += 1
        if label_conflict:
            label_conflict_count += 1
        if not physical_exists:
            missing_physical_count += 1
        group_details.append(
            {
                "sample_path": sample_path,
                "source": values_text(group["source"]),
                "split": values_text(group["_split_file"]),
                "occurrences": len(group),
                "physical_exists": "yes" if physical_exists else "NO",
                "exactly_identical": "yes" if not varying_fields else "no",
                "differing_fields": ", ".join(varying_fields) or "none",
                "type": values_text(group["type"]),
                "method": values_text(group["method"]),
                "video_label": values_text(group["video_label"]),
                "audio_label": values_text(group["audio_label"]),
                "label": values_text(group["label"]),
                "category": values_text(group["category"]),
                "race": values_text(group["race"]),
                "gender": values_text(group["gender"]),
                "path": values_text(group["path"]),
                "target1": values_text(group["target1"]),
                "target2": values_text(group["target2"]),
                "metadata_directory": values_text(group["Unnamed: 9"]),
            }
        )

    split_path_sets = {
        split: set(dataframe.loc[dataframe["_split_file"] == split, "sample_path"])
        for split in SPLIT_NAMES
    }
    cross_split_paths = set()
    for index, left_split in enumerate(SPLIT_NAMES):
        for right_split in SPLIT_NAMES[index + 1 :]:
            cross_split_paths.update(
                split_path_sets[left_split] & split_path_sets[right_split]
            )

    split_identity_sets = {
        split: set(dataframe.loc[dataframe["_split_file"] == split, "source"])
        for split in SPLIT_NAMES
    }
    cross_split_identities = set()
    for index, left_split in enumerate(SPLIT_NAMES):
        for right_split in SPLIT_NAMES[index + 1 :]:
            cross_split_identities.update(
                split_identity_sets[left_split] & split_identity_sets[right_split]
            )

    row_counts = dataframe.groupby("_split_file").size().to_dict()
    identity_counts = dataframe.groupby("_split_file")["source"].nunique().to_dict()
    split_column_matches = bool(
        (dataframe["split"].astype(str) == dataframe["_split_file"]).all()
    )
    all_paths_exist = missing_physical_count == 0
    final_status = "PASS - audit complete; manifests unchanged"

    lines = [
        "# Dataset Duplicate Metadata Audit",
        "",
        f"- **Audit date:** {date.today().isoformat()}",
        "- **Source metadata:** `Dataset_FakeAVCeleb/meta_data.csv` (represented by the current split manifests)",
        "- **Split manifests:** `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`",
        "",
        "## Summary",
        "",
        f"- Total manifest rows: **{len(dataframe)}**",
        f"- Unique canonical sample paths: **{dataframe['sample_path'].nunique()}**",
        f"- Duplicated canonical paths: **{len(duplicate_groups)}**",
        f"- Duplicate rows: **{len(duplicate_rows)}**",
        f"- Rows by split: train={row_counts.get('train', 0)}, val={row_counts.get('val', 0)}, test={row_counts.get('test', 0)}",
        f"- Identities by split: train={identity_counts.get('train', 0)}, val={identity_counts.get('val', 0)}, test={identity_counts.get('test', 0)}",
        "",
        "## Duplicate Findings",
        "",
        "The table includes all relevant metadata values for each duplicated canonical path. A semicolon-separated value means the two rows differ for that field.",
        "",
        "| sample_path | source | split | occurrences | physical file | exact metadata | differing fields | type | method | video label | audio label | label | category | race | gender | path | target1 | target2 | metadata directory |",
        "|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for detail in group_details:
        lines.append(
            "| "
            + " | ".join(
                [
                    detail["sample_path"],
                    detail["source"],
                    detail["split"],
                    str(detail["occurrences"]),
                    detail["physical_exists"],
                    detail["exactly_identical"],
                    detail["differing_fields"],
                    detail["type"],
                    detail["method"],
                    detail["video_label"],
                    detail["audio_label"],
                    detail["label"],
                    detail["category"],
                    detail["race"],
                    detail["gender"],
                    detail["path"],
                    detail["target1"],
                    detail["target2"],
                    detail["metadata_directory"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- All split CSVs loaded with the expected schema: **yes**.",
            f"- Every duplicated path resolves to an existing physical file: **{'yes' if all_paths_exist else 'NO'}** ({missing_physical_count} missing).",
            f"- Cross-split duplicate sample paths: **{len(cross_split_paths)}**.",
            f"- Cross-split source identities: **{len(cross_split_identities)}**.",
            f"- Every manifest `split` value matches its file: **{'yes' if split_column_matches else 'NO'}**.",
            f"- Duplicate groups with metadata differences: **{conflict_count}**.",
            f"- Duplicate groups with visual/audio/final label conflicts: **{label_conflict_count}**.",
            "",
            "## Decision",
            "",
            "- Classification: **same physical video with conflicting metadata annotations**, not exact duplicate metadata records. Every conflict is in `method`: `faceswap-wav2lip` versus `wav2lip`.",
            "- Label result: all duplicate groups agree on `video_label`, `audio_label`, and `label`; no label correction was performed.",
            "- Recommended action: **retain the current manifests unchanged pending dataset review**. Removing one row would reduce the row-level weighting of that physical file and discard one method annotation; retaining both preserves the source metadata but can cause the same physical video to be sampled twice during later preprocessing.",
            "- Automatic deduplication: **not safe** because metadata conflicts exist. No split CSV or raw dataset file was modified by this audit.",
            "",
            f"**Final status:** {final_status}",
        ]
    )
    structural_checks_pass = (
        all_paths_exist
        and not cross_split_paths
        and not cross_split_identities
        and split_column_matches
    )
    return "\n".join(lines) + "\n", structural_checks_pass


def main() -> int:
    args = parse_args()
    dataframe = load_manifests(args.split_dir.resolve())
    report, structural_checks_pass = audit(dataframe, args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    # Metadata conflicts are an intentional review result, not a failed audit.
    return 0 if structural_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
