"""Input validation and the explicit, versioned candidate feature map."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_mapping(path: Path) -> dict:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    features = mapping["features"]
    for group, names in mapping["groups"].items():
        if not names or len(set(names)) != len(names) or any(n not in features for n in names):
            raise ValueError(f"Invalid feature group: {group}")
    return mapping


def columns_for(mapping: dict, group: str, domain: str) -> list[str]:
    try:
        return [mapping["features"][name][domain] for name in mapping["groups"][group]]
    except KeyError as exc:
        raise ValueError(f"Unknown group/domain/mapping entry: {group}/{domain}/{exc}") from exc


def load_features(path: Path, mapping: dict, group: str, domain: str,
                  *, with_labels: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load only selected columns. Never derive a feature from label or target data."""
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = columns_for(mapping, group, domain)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    expected = columns + (["Label"] if with_labels else [])
    missing = [name for name in expected if name not in header]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; first columns: {header[:12]}")
    frame = pd.read_csv(path, usecols=expected, low_memory=False)
    # usecols preserves file order; set the canonical order explicitly.
    x = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x.columns = mapping["groups"][group]
    if x.empty:
        raise ValueError(f"Empty CSV: {path}")
    if not with_labels:
        return x, None
    numeric = pd.to_numeric(frame["Label"], errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any():
        raise ValueError(f"{path}: Label must be numeric, with 0=benign and positive=attack")
    y = (numeric.to_numpy() != 0).astype(np.int8)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{path}: binary evaluation requires both benign and attack")
    return x, y


def overlap_report(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Feature-vector overlap only; identical vectors need not be the same event.

    Labels are deliberately excluded. This does not test device/time separation.
    """
    if list(left.columns) != list(right.columns):
        raise ValueError("Overlap audit requires the same ordered feature columns")
    # Normalize dtypes so integer and float representations compare equally.
    a = pd.MultiIndex.from_frame(left.astype("float64"))
    b = pd.MultiIndex.from_frame(right.astype("float64"))
    return {"left_rows": len(left), "right_rows": len(right),
            "left_duplicate_vectors": int(a.duplicated().sum()),
            "right_duplicate_vectors": int(b.duplicated().sum()),
            "left_rows_matching_right": int(a.isin(b).sum()),
            "right_rows_matching_left": int(b.isin(a).sum()),
            "note": "Identical selected features are not proof of duplicated events; device/time independence is unverified."}


def impute_from_train(train: pd.DataFrame, *others: pd.DataFrame):
    """Impute with source TRAIN medians only; all values remain in shared units."""
    medians = train.median(axis=0).fillna(0.0)
    def convert(frame):
        return frame.fillna(medians).to_numpy(dtype=np.float32)
    return convert(train), [convert(frame) for frame in others], medians.to_dict()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def claim_final_evaluation(target_csv: Path, record: dict) -> Path:
    """Reserve one FINAL attempt across both CLI entry points and result folders."""
    path = target_csv.parent / (".final_evaluation_" + sha256(target_csv) + ".json")
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({**record, "status": "attempted"}, stream, indent=2)
    except FileExistsError as exc:
        raise ValueError("FINAL evaluation already attempted for this dataset; review its audit record") from exc
    return path
