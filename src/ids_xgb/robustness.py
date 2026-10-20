"""E6 analogue: deterministic, label-free measurement-drift stress test.

This is not FGSM/PGD or a demonstrated realizable network attack.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .data import load_features, load_mapping, sha256
from .experiment import artifact_name, metrics, save_json


def perturb_features(frame: pd.DataFrame, epsilon: float, direction: str) -> pd.DataFrame:
    if not 0 <= epsilon <= 1 or direction not in {"up", "down"}:
        raise ValueError("epsilon must be in [0, 1]; direction must be up or down")
    scale = 1 + epsilon if direction == "up" else 1 - epsilon
    x = frame.copy()
    # Scale all rates by one factor: flow ~= forward + backward remains true.
    # Scale all length statistics by one factor: min <= mean <= max remains true.
    for col in x:
        if col.endswith("_per_second") or col.startswith("packet_length_"):
            x[col] = x[col].clip(lower=0).mul(scale)
    return x


def run_drift(run_dir: Path, iot_dev: Path, mapping_path: Path,
              epsilons: list[float], output: Path) -> list[dict]:
    if output.exists():
        raise FileExistsError(output)
    if not epsilons or any(e < 0 or e > 1 for e in epsilons):
        raise ValueError("epsilons must lie in [0, 1]")
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    mapping = load_mapping(mapping_path)
    if sha256(mapping_path) != meta["mapping_sha256"]:
        raise ValueError("Mapping changed since training")
    original, y = load_features(iot_dev, mapping, meta["group"], "iot")
    filename = artifact_name(meta)
    if meta.get("model_type", "xgboost") == "xgboost":
        model = XGBClassifier()
        model.load_model(str(run_dir / filename))
    else:
        model = joblib.load(run_dir / filename)  # Only open locally produced runs.
    rows = []
    for epsilon in epsilons:
        for direction in (["up"] if epsilon == 0 else ["up", "down"]):
            perturbed = perturb_features(original, epsilon, direction)
            values = perturbed.fillna(pd.Series(meta["medians"])).to_numpy(np.float32)
            score = model.predict_proba(values)[:, 1]
            m = metrics(y, score, meta["threshold"], meta["source_validation_margin_for_90pct_acceptance"])
            rows.append({"epsilon": float(epsilon), "direction": direction,
                         "macro_f1": m["macro_f1"], "auroc": m["auroc"],
                         "fpr": m["fpr"], "fnr": m["fnr"],
                         "accepted_fraction": m["accepted_fraction"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"source": meta["source"], "model": meta.get("model_type", "xgboost"),
              "group": meta["group"], "target_sha256": sha256(iot_dev),
              "model_sha256": sha256(run_dir / filename), "rows": rows,
              "note": "Synthetic multiplicative feature-measurement drift on IoT DEV; no target labels used to choose perturbations. Not FGSM/PGD and not a proven realizable packet attack."}
    save_json(output, report)
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    return rows
