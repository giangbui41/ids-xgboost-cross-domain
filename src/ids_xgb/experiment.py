"""Source-only training, frozen model evaluation, and diagnostic abstention."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .data import claim_final_evaluation, impute_from_train, load_features, load_mapping, sha256


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float,
            abstain_margin: float = 0.0, *, fp_cost: float = 100.0,
            fn_cost: float = 1.0) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics require equally sized vectors and both binary classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities within [0,1]")
    if not 0 <= threshold <= 1 or not 0 <= abstain_margin <= 1:
        raise ValueError("Threshold and abstention margin must lie within [0,1]")
    if not np.isfinite([fp_cost, fn_cost]).all() or fp_cost < 0 or fn_cost < 0 or fp_cost + fn_cost <= 0:
        raise ValueError("Costs must be finite, nonnegative, and not both zero")
    predicted = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    keep = np.abs(scores - threshold) >= abstain_margin
    selected = None
    if keep.any() and not keep.all() and len(np.unique(y[keep])) == 2:
        selected = {
            "macro_f1": float(f1_score(y[keep], predicted[keep], average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y[keep], predicted[keep])),
            "fpr": float(np.mean(predicted[keep][y[keep] == 0] == 1)),
            "fnr": float(np.mean(predicted[keep][y[keep] == 1] == 0)),
        }
    return {
        "n": len(y), "attack_prevalence": float(y.mean()),
        "threshold": float(threshold), "macro_f1": float(f1_score(y, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "fpr": float(fp / (fp + tn)), "fnr": float(fn / (fn + tp)),
        "predicted_attack_fraction": float(predicted.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fp_cost": float(fp_cost), "fn_cost": float(fn_cost),
        "total_cost": float(fp_cost * fp + fn_cost * fn),
        "cost_per_sample": float((fp_cost * fp + fn_cost * fn) / len(y)),
        "always_benign_cost": float(fn_cost * (fn + tp)),
        "always_attack_cost": float(fp_cost * (tn + fp)),
        "accepted_fraction": float(keep.mean()), "accepted_n": int(keep.sum()),
        "accepted_metrics": selected,
    }


def metrics_with_cost(y, scores, threshold, cost_fp=100, cost_fn=1):
    """Guide-compatible cost metrics while retaining the original metrics API."""
    result = metrics(y, scores, threshold, fp_cost=cost_fp, fn_cost=cost_fn)
    return {**result, "false_alarm_rate": result["fpr"]}


metrics_cost_sensitive = metrics_with_cost


def select_cost_threshold(y: np.ndarray, scores: np.ndarray, *, fp_cost: float = 100.0,
                          fn_cost: float = 1.0) -> float:
    """Minimize empirical cost on an explicitly reserved calibration set.

    Include all-benign/all-attack policies; ties prefer fewer alerts. Scores at
    one are clipped only for this search, consistently with the E8 predictor.
    Never call this with evaluation labels.
    """
    metrics(y, scores, .5, fp_cost=fp_cost, fn_cost=fn_cost)
    scores = np.minimum(np.asarray(scores, dtype=float), np.nextafter(1.0, 0.0))
    y = np.asarray(y)
    order = np.argsort(-scores, kind="stable")
    ordered_scores, labels = scores[order], y[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    ends = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), len(y) - 1]
    costs = np.r_[fn_cost * np.sum(y == 1),
                  fp_cost * fp[ends] + fn_cost * (np.sum(y == 1) - tp[ends])]
    thresholds = np.r_[1.0, ordered_scores[ends]]
    return float(thresholds[int(np.argmin(costs))])


def select_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    # Fixed search grid; the IoT labels never influence it.
    grid = np.linspace(0.005, 0.995, 199)
    values = [f1_score(y, scores >= t, average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(values))])


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


MODEL_TYPES = ("xgboost", "random_forest", "logistic_regression")


def make_model(model_type: str, seed: int, n_jobs: int):
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=n_jobs, random_state=seed,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=16,
                                      min_samples_leaf=2, n_jobs=n_jobs, random_state=seed)
    if model_type == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                                                  random_state=seed))
    raise ValueError(f"Unknown model: {model_type}")


def artifact_name(metadata: dict) -> str:
    return "model.json" if metadata.get("model_type", "xgboost") == "xgboost" else "model.joblib"


def train(source: str, source_csv: Path, mapping_path: Path, group: str,
          output_dir: Path, seed: int = 42, n_jobs: int = 4,
          model_type: str = "xgboost") -> dict:
    if source not in {"2017", "2018"}:
        raise ValueError("source must be 2017 or 2018")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    mapping = load_mapping(mapping_path)
    x, y = load_features(source_csv, mapping, group, source)
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    tr, va = train_test_split(trainval, test_size=0.125, stratify=y[trainval], random_state=seed)
    a, (b, c), medians = impute_from_train(x.iloc[tr], x.iloc[va], x.iloc[test])
    model = make_model(model_type, seed, n_jobs)
    weights = compute_sample_weight(class_weight="balanced", y=y[tr])
    if model_type == "logistic_regression":
        model.fit(a, y[tr], logisticregression__sample_weight=weights)
    else:
        model.fit(a, y[tr], sample_weight=weights)
    val_scores = model.predict_proba(b)[:, 1]
    threshold = select_threshold(y[va], val_scores)
    # Experimental rejection rule: source-validation margin for 90% acceptance.
    # It has no statistical coverage guarantee under domain shift.
    margin = float(np.quantile(np.abs(val_scores - threshold), 0.10))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = "model.json" if model_type == "xgboost" else "model.joblib"
    if model_type == "xgboost":
        model.save_model(str(output_dir / model_file))
    else:
        joblib.dump(model, output_dir / model_file)
    metadata = {
        "protocol": "source-only-70-10-20-v1", "source": source, "group": group,
        "model_type": model_type, "model_file": model_file,
        "features": mapping["groups"][group], "mapping_version": mapping["mapping_version"],
        "mapping_sha256": sha256(mapping_path), "source_sha256": sha256(source_csv),
        "seed": seed, "n_jobs": n_jobs, "n_train": len(tr), "n_validation": len(va),
        "n_source_test": len(test), "medians": medians,
        "threshold": threshold, "source_validation_margin_for_90pct_acceptance": margin,
        "source_validation": metrics(y[va], val_scores, threshold, margin),
        "source_test": metrics(y[test], model.predict_proba(c)[:, 1], threshold, margin),
        "note": "Mapping is a semantic hypothesis. No target labels were read in training.",
    }
    save_json(output_dir / "run.json", metadata)
    return metadata


def freeze(run_dir: Path, frozen_dir: Path) -> dict:
    if frozen_dir.exists():
        raise FileExistsError(f"Frozen directory already exists: {frozen_dir}")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    frozen_dir.mkdir(parents=True)
    filename = artifact_name(metadata)
    shutil.copy2(run_dir / filename, frozen_dir / filename)
    shutil.copy2(run_dir / "run.json", frozen_dir / "run.json")
    lock = {"model_sha256": sha256(frozen_dir / filename),
            "run_sha256": sha256(frozen_dir / "run.json"),
            "protocol": metadata["protocol"]}
    save_json(frozen_dir / "frozen.json", lock)
    return lock


def evaluate(model_dir: Path, target_csv: Path, mapping_path: Path,
             stage: str, output_json: Path) -> dict:
    if stage not in {"dev", "final"}:
        raise ValueError("stage must be dev or final")
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_json}")
    metadata = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    filename = artifact_name(metadata)
    mapping = load_mapping(mapping_path)
    if sha256(mapping_path) != metadata["mapping_sha256"]:
        raise ValueError("Feature mapping changed since training")
    if stage == "final":
        lock = json.loads((model_dir / "frozen.json").read_text(encoding="utf-8"))
        if lock["model_sha256"] != sha256(model_dir / filename) or lock["run_sha256"] != sha256(model_dir / "run.json"):
            raise ValueError("Frozen model or protocol changed")
        claim_final_evaluation(target_csv, {"model_dir": str(model_dir.resolve()),
                                            "output": str(output_json.resolve()), "selection": lock})
    x, y = load_features(target_csv, mapping, metadata["group"], "iot")
    x = x.fillna(pd.Series(metadata["medians"])).to_numpy(dtype=np.float32)
    if metadata.get("model_type", "xgboost") == "xgboost":
        model = XGBClassifier()
        model.load_model(str(model_dir / filename))
    else:
        # Load only local model files produced by this project; joblib uses pickle.
        model = joblib.load(model_dir / filename)
    score = model.predict_proba(x)[:, 1]
    result = {
        "stage": stage, "source": metadata["source"], "group": metadata["group"],
        "seed": metadata["seed"], "model_type": metadata.get("model_type", "xgboost"),
        "protocol": metadata["protocol"], "mapping_sha256": metadata["mapping_sha256"],
        "model_sha256": sha256(model_dir / filename),
        "target_sha256": sha256(target_csv),
        "metrics": metrics(y, score, metadata["threshold"], metadata["source_validation_margin_for_90pct_acceptance"]),
        "note": "DEV is diagnostic. FINAL must be evaluated only after decisions and model are frozen. Abstention is exploratory and has no cross-domain guarantee.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_json, result)
    return result
