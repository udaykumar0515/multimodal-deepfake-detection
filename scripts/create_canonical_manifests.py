"""Create one-row-per-physical-video manifests from identity split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {
    "source", "target1", "target2", "method", "category", "type", "race",
    "gender", "path", "Unnamed: 9", "video_label", "audio_label", "label",
    "sample_path", "split",
}
INVARIANT_COLUMNS = (
    "source", "target1", "target2", "category", "type", "race", "gender",
    "path", "Unnamed: 9", "video_label", "audio_label", "label", "split",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/canonical"))
    return parser.parse_args()


def load_split(path: Path, split: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not (dataframe["split"].astype(str) == split).all():
        raise ValueError(f"{path} contains rows with an incorrect split value")
    return dataframe


def provenance_records(group: pd.DataFrame) -> str:
    records = group.sort_values(["method", "target1", "target2"])[
        [
            "source", "target1", "target2", "method", "category", "type",
            "race", "gender", "path", "Unnamed: 9",
        ]
    ].rename(columns={"Unnamed: 9": "metadata_directory"}).astype(str).to_dict("records")
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def provenance_record(row: pd.Series) -> str:
    record = {
        "source": str(row["source"]),
        "target1": str(row["target1"]),
        "target2": str(row["target2"]),
        "method": str(row["method"]),
        "category": str(row["category"]),
        "type": str(row["type"]),
        "race": str(row["race"]),
        "gender": str(row["gender"]),
        "path": str(row["path"]),
        "metadata_directory": str(row["Unnamed: 9"]),
    }
    return json.dumps([record], sort_keys=True, separators=(",", ":"))


def canonicalize(dataframe: pd.DataFrame) -> pd.DataFrame:
    grouped = dataframe.groupby("sample_path", sort=True)
    for column in INVARIANT_COLUMNS:
        conflicts = grouped[column].nunique(dropna=False)
        if conflicts.gt(1).any():
            sample_path = conflicts[conflicts.gt(1)].index[0]
            raise ValueError(f"Conflicting {column} values for canonical path {sample_path}")
    result = grouped.first().reset_index()
    result["metadata_row_count"] = grouped.size().reindex(result["sample_path"]).to_numpy()
    result["method_annotations"] = grouped["method"].agg(
        lambda values: "|".join(sorted(values.astype(str).unique()))
    ).reindex(result["sample_path"]).to_numpy()
    duplicate_paths = set(dataframe.loc[dataframe["sample_path"].duplicated(keep=False), "sample_path"])
    provenance = {
        sample_path: provenance_records(group)
        for sample_path, group in dataframe[dataframe["sample_path"].isin(duplicate_paths)].groupby("sample_path", sort=True)
    }
    result["metadata_provenance"] = [
        provenance.get(sample_path, provenance_record(row))
        for sample_path, (_, row) in zip(
            result["sample_path"], result.iterrows()
        )
    ]
    columns = list(INVARIANT_COLUMNS) + [
        "sample_path", "metadata_row_count", "method_annotations", "metadata_provenance",
    ]
    return result[columns].sort_values("sample_path").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        source = load_split(args.input_dir / f"{split}.csv", split)
        canonical = canonicalize(source)
        canonical.to_csv(args.output_dir / f"{split}.csv", index=False)
        print(
            f"{split}: metadata_rows={len(source)} canonical_videos={len(canonical)} "
            f"duplicate_rows_collapsed={len(source) - len(canonical)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())