"""Create and verify reproducible identity-based FakeAVCeleb splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "source",
    "type",
    "path",
    "Unnamed: 9",
}
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create verified identity-based FakeAVCeleb train/val/test CSV files."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("Dataset_FakeAVCeleb/meta_data.csv"),
        help="Path to the FakeAVCeleb metadata CSV.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset_FakeAVCeleb"),
        help="Local FakeAVCeleb directory used to validate sample references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory where train.csv, val.csv, and test.csv are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for identity assignment (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def canonical_relative_path(metadata_path: pd.Series, filename: pd.Series) -> pd.Series:
    directories = (
        metadata_path.astype(str)
        .str.replace("\\\\", "/", regex=False)
        .str.replace(r"^FakeAVCeleb/", "", regex=True)
        .str.rstrip("/")
    )
    filenames = filename.astype(str).str.strip().str.replace("\\\\", "/", regex=False)
    return directories + "/" + filenames


def add_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result["video_label"] = result["type"].str.startswith("FakeVideo-").map(
        {True: "Fake", False: "Real"}
    )
    result["audio_label"] = result["type"].str.endswith("-FakeAudio").map(
        {True: "Fake", False: "Real"}
    )
    # The project blueprint's primary visual label is the video component.
    result["label"] = result["video_label"]
    return result


def assign_identities(identities: Iterable[str], seed: int) -> dict[str, str]:
    ordered = pd.Series(sorted(identities), dtype="string")
    shuffled = ordered.sample(frac=1, random_state=seed).tolist()
    identity_count = len(shuffled)
    train_count = int(identity_count * 0.70)
    val_count = int(identity_count * 0.15)
    assignments: dict[str, str] = {}
    for split, split_identities in (
        ("train", shuffled[:train_count]),
        ("val", shuffled[train_count : train_count + val_count]),
        ("test", shuffled[train_count + val_count :]),
    ):
        assignments.update({identity: split for identity in split_identities})
    return assignments


def verify_splits(
    splits: dict[str, pd.DataFrame],
    all_rows: pd.DataFrame,
    dataset_root: Path,
    identity_assignments: dict[str, str],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    split_identity_sets = {
        split: set(dataframe["source"].astype(str))
        for split, dataframe in splits.items()
    }

    for left_index, left_split in enumerate(SPLIT_NAMES):
        for right_split in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_identity_sets[left_split] & split_identity_sets[right_split]
            print(f"Identity overlap {left_split} & {right_split}: {len(overlap)}")
            if overlap:
                failures.append(f"identity overlap between {left_split} and {right_split}")

    assigned_identities = set().union(*split_identity_sets.values())
    expected_identities = set(identity_assignments)
    if assigned_identities != expected_identities:
        failures.append("not every source identity was assigned to a split")

    split_row_count = sum(len(dataframe) for dataframe in splits.values())
    if split_row_count != len(all_rows):
        failures.append("split row counts do not equal metadata row count")

    all_sample_paths: dict[str, set[str]] = {}
    for split, dataframe in splits.items():
        paths = set(dataframe["sample_path"])
        all_sample_paths[split] = paths
        missing = [
            sample_path
            for sample_path in paths
            if not (dataset_root / sample_path).is_file()
        ]
        duplicate_rows = int(dataframe["sample_path"].duplicated(keep=False).sum())
        print(
            f"{split.title()} sample paths: {len(paths)} unique "
            f"({duplicate_rows} duplicate metadata rows)"
        )
        if missing:
            failures.append(f"{split} contains {len(missing)} missing sample paths")

    cross_split_duplicates: set[str] = set()
    for left_index, left_split in enumerate(SPLIT_NAMES):
        for right_split in SPLIT_NAMES[left_index + 1 :]:
            cross_split_duplicates.update(
                all_sample_paths[left_split] & all_sample_paths[right_split]
            )
    print(f"Duplicate sample paths across splits: {len(cross_split_duplicates)}")
    if cross_split_duplicates:
        failures.append("sample paths occur in more than one split")

    source_counts = all_rows["source"].value_counts()
    for split, dataframe in splits.items():
        for identity in dataframe["source"].unique():
            if len(dataframe[dataframe["source"] == identity]) != source_counts[identity]:
                failures.append(f"identity rows were divided or lost for {identity}")
                break

    if all_rows["sample_path"].duplicated().any():
        duplicate_count = int(all_rows["sample_path"].duplicated(keep=False).sum())
        print(f"Duplicate metadata rows for a sample path: {duplicate_count}")

    return not failures, failures


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir

    dataframe = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    if dataframe["source"].isna().any():
        raise ValueError("Metadata contains missing source identities")

    dataframe = add_labels(dataframe)
    dataframe["sample_path"] = canonical_relative_path(
        dataframe["Unnamed: 9"], dataframe["path"]
    )
    identity_assignments = assign_identities(dataframe["source"].unique(), args.seed)
    dataframe["split"] = dataframe["source"].map(identity_assignments)
    if dataframe["split"].isna().any():
        raise ValueError("One or more metadata rows has no identity split assignment")

    splits = {
        split: dataframe[dataframe["split"] == split].copy()
        for split in SPLIT_NAMES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_dataframe in splits.items():
        split_dataframe.to_csv(output_dir / f"{split}.csv", index=False)

    print("Identity Split Verification")
    print("---------------------------")
    print(f"Metadata rows: {len(dataframe)}")
    print(f"Total identities: {dataframe['source'].nunique()}")
    for split in SPLIT_NAMES:
        split_dataframe = splits[split]
        print(
            f"{split.title()} identities: {split_dataframe['source'].nunique()} | "
            f"samples: {len(split_dataframe)} | "
            f"Real: {(split_dataframe['label'] == 'Real').sum()} | "
            f"Fake: {(split_dataframe['label'] == 'Fake').sum()}"
        )

    passed, failures = verify_splits(
        splits, dataframe, dataset_root, identity_assignments
    )
    print(f"Random seed: {args.seed}")
    print(f"Final status: {'PASS' if passed else 'FAIL'}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
