"""Optional fresh preparation from released CSVs; never modifies input files."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_mapping, sha256
from .experiment import save_json


def load_and_engineer_features(path, mapping, group, source):
    """Load mapped features and apply the guide's deterministic row transforms."""
    from .data import load_features
    from .features import engineer_features
    x, y = load_features(path, mapping, group, source)
    return engineer_features(x), y


def normal(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def discover_header(path: Path, mapping: dict, domain: str) -> tuple[list[str], str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    normalized = {}
    for name in header:
        key = normal(name)
        if key in normalized:
            raise ValueError(f"{path}: ambiguous normalized column {name}")
        normalized[key] = name
    label = normalized.get("label")
    if label is None:
        raise ValueError(f"{path}: no Label column")
    result = []
    for feature in mapping["groups"]["seven"]:
        candidate = mapping["features"][feature][domain]
        key = normal(candidate)
        if key not in normalized:
            raise ValueError(f"{path}: missing {candidate} ({feature})")
        result.append(normalized[key])
    return result, label


def binary_label(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("Missing labels in raw dataset")
    text = series.astype(str).str.strip().str.lower()
    if text.eq("").any():
        raise ValueError("Missing labels in raw dataset")
    numeric = pd.to_numeric(text, errors="coerce")
    is_numeric = numeric.notna()
    if (numeric[is_numeric] < 0).any() or not np.isfinite(numeric[is_numeric]).all():
        raise ValueError("Numeric labels must be finite and nonnegative")
    if text.isin(["nan", "none", "null"]).any():
        raise ValueError("Missing labels in raw dataset")
    result = (~text.isin(["benign", "normal"])).astype(np.int8)
    result.loc[is_numeric] = numeric.loc[is_numeric].ne(0).astype(np.int8)
    return result


def sample_chunks(paths: list[Path], mapping: dict, domain: str,
                  fraction: float, seed: int, *, keep_all_numeric: bool):
    if not (0 < fraction <= 1):
        raise ValueError("sample fraction must be in (0, 1]")
    frames = []
    statistics = []
    for file_index, path in enumerate(paths):
        columns, label = discover_header(path, mapping, domain)
        chosen = None if keep_all_numeric else set(columns + [label])
        sampled = []
        read_rows = 0
        for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=chosen,
                                                        chunksize=200_000, low_memory=False)):
            read_rows += len(chunk)
            if fraction < 1:
                count = max(1, round(len(chunk) * fraction))
                chunk = chunk.sample(n=min(count, len(chunk)),
                                     random_state=seed + 1000 * file_index + chunk_index)
            selected = chunk if keep_all_numeric else chunk.loc[:, columns + [label]].copy()
            sampled.append(selected)
        if not sampled:
            continue
        frame = pd.concat(sampled, ignore_index=True)
        labels = binary_label(frame[label])
        if keep_all_numeric:
            numeric = frame.drop(columns=label).select_dtypes(include=[np.number]).copy()
            for col in columns:
                numeric[col] = pd.to_numeric(frame[col], errors="coerce")
            numeric = numeric.rename(columns=dict(zip(columns, [
                mapping["features"][f][domain] for f in mapping["groups"]["seven"]])))
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            valid = ~numeric.isna().any(axis=1)
            output = numeric.loc[valid].reset_index(drop=True)
            output["Label"] = labels.loc[valid].reset_index(drop=True)
        else:
            output = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
            output = output.replace([np.inf, -np.inf], np.nan)
            output.columns = [mapping["features"][f][domain]
                              for f in mapping["groups"]["seven"]]
            output["Label"] = labels.to_numpy()
        frames.append(output)
        statistics.append({"file": str(path), "read_rows": read_rows,
                           "sampled_rows": len(output)})
    if not frames:
        raise ValueError("No rows prepared")
    return pd.concat(frames, ignore_index=True), statistics


def prepare_source(raw_dir: Path, output: Path, mapping_path: Path,
                   domain: str, fraction: float, seed: int) -> dict:
    if domain not in {"2017", "2018"}:
        raise ValueError("source must be 2017 or 2018")
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSVs in {raw_dir}")
    if output.exists():
        raise FileExistsError(output)
    mapping = load_mapping(mapping_path)
    frame, files_info = sample_chunks(files, mapping, domain, fraction, seed,
                                      keep_all_numeric=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    report = {"domain": domain, "rows": len(frame), "attack_prevalence": float(frame.Label.mean()),
              "sample_frac_per_chunk": fraction, "seed": seed, "files": files_info,
              "mapping_sha256": sha256(mapping_path), "output_sha256": sha256(output),
              "note": "No balancing, scaling or test-set fitting. Keep original CSVs and this manifest."}
    save_json(output.with_suffix(output.suffix + ".meta.json"), report)
    return report


def prepare_iot(raw_dir: Path, dev_output: Path, final_output: Path,
                mapping_path: Path, fraction: float, seed: int, *, deduplicate: bool = False) -> dict:
    if dev_output.resolve() == final_output.resolve():
        raise ValueError("DEV and FINAL output paths must differ")
    if dev_output.exists() or final_output.exists():
        raise FileExistsError("Refusing to overwrite existing IoT split")
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSVs in {raw_dir}")
    mapping = load_mapping(mapping_path)
    pool, files_info = sample_chunks(files, mapping, "iot", fraction, seed,
                                     keep_all_numeric=True)
    sampled_rows = len(pool)
    feature_columns = [c for c in pool if c != "Label"]
    duplicate_vectors = int(pool.duplicated(subset=feature_columns).sum())
    if deduplicate:
        unique_rows = pool.drop_duplicates()
        if unique_rows.duplicated(subset=feature_columns).any():
            raise ValueError("Identical feature vectors have conflicting labels; resolve before deduplication")
        pool = unique_rows.reset_index(drop=True)
    index = np.arange(len(pool))
    dev_idx, final_idx = train_test_split(index, test_size=0.20,
                                          stratify=pool["Label"], random_state=seed)
    dev_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    pool.iloc[dev_idx].reset_index(drop=True).to_csv(dev_output, index=False)
    pool.iloc[final_idx].reset_index(drop=True).to_csv(final_output, index=False)
    report = {"domain": "iot", "source_rows": len(pool), "sampled_rows": sampled_rows,
              "duplicate_feature_vectors": duplicate_vectors, "deduplicated": deduplicate,
              "rows_removed": sampled_rows - len(pool), "dev_rows": len(dev_idx),
              "final_rows": len(final_idx), "dev_attack_prevalence": float(pool.Label.iloc[dev_idx].mean()),
              "final_attack_prevalence": float(pool.Label.iloc[final_idx].mean()),
              "sample_frac_per_chunk": fraction, "seed": seed, "files": files_info,
              "dev_sha256": sha256(dev_output), "final_sha256": sha256(final_output),
              "note": "Random stratified split after sampling across raw CSVs. It is not an independent device/time holdout. Never select models on FINAL."}
    save_json(dev_output.parent / "iot_split_manifest.json", report)
    return report
