"""Budgeted, labeled-target adaptation. IoT FINAL_TEST is never opened here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .data import load_features, load_mapping
from .experiment import make_model, metrics
from .matrix import DATA_FILES, split_source
from .robustness import perturb_features

METHODS = ("source_only", "target_only", "pooled", "weighted")


def improvement_config(config: dict) -> dict:
    """Translate the improved E4 method list into the shared E4/E8 runner.

    Legacy E4 configs without feature_engineering retain their original path.
    New comparisons are explicit variants so calibration/TTA never silently
    replace an unadapted baseline. Hybrid is intentionally not an E4 method.
    """
    options = dict(config)
    if options.pop("model", "xgboost") != "xgboost":
        raise ValueError("Improved E4 currently requires xgboost")
    enabled = options.pop("feature_engineering", True)
    compare_raw = options.pop("compare_raw", True)
    use_tta = options.pop("use_tta", False)
    tta_method = options.pop("tta_method", "entropy")
    if any(type(v) is not bool for v in (enabled, compare_raw, use_tta)):
        raise ValueError("E4 feature_engineering, compare_raw and use_tta must be boolean")
    if tta_method not in {"entropy", "self_training"}:
        raise ValueError("Unknown E4 tta_method")
    if "variants" in options:
        raise ValueError("E4 uses methods; use E8 for custom variants")
    methods = options.pop("methods", [])
    allowed = {*METHODS, "weighted_adaptive", "self_training"}
    if (not isinstance(methods, list) or not methods or any(not isinstance(m, str) or m not in allowed for m in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError("Invalid improved E4 methods")
    calibration = dict(options.get("calibration", {}))
    do_calibration = calibration.pop("enabled", False)
    if type(do_calibration) is not bool:
        raise ValueError("calibration.enabled must be boolean")
    options["calibration"] = calibration
    variants = [{"name": m, "method": m, "feature_engineering": enabled} for m in methods]
    if compare_raw and enabled:
        if "weighted" not in methods:
            raise ValueError("compare_raw requires weighted in E4 methods")
        variants.append({"name": "weighted_raw", "method": "weighted", "feature_engineering": False})
    if do_calibration:
        if "weighted" not in methods:
            raise ValueError("E4 calibration comparison requires weighted")
        variants.append({"name": "weighted_calibrated", "method": "weighted",
                         "feature_engineering": enabled, "calibrate": True})
    if use_tta:
        if "weighted" not in methods:
            raise ValueError("E4 TTA comparison requires weighted")
        variants.append({"name": "weighted_tta_" + tta_method, "method": "weighted",
                         "feature_engineering": enabled, "tta": tta_method})
    options["variants"] = variants
    return options


def run_fewshot_improvements(config, data_dir, mapping_path, work):
    from .improvements import run_improvements
    return run_improvements(improvement_config(config), data_dir, mapping_path, work)


@dataclass
class FewshotData:
    x_source: pd.DataFrame
    y_source: np.ndarray
    x_adapt: pd.DataFrame
    y_adapt: np.ndarray
    x_holdout: pd.DataFrame
    y_holdout: np.ndarray
    source_train_indices: np.ndarray
    adapt_indices: np.ndarray
    holdout_indices: np.ndarray
    seed: int

    def target_sample(self, k: int):
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer")
        selected = []
        # Fixed class-specific permutations make budgets nested across k.
        for label in (0, 1):
            candidates = np.flatnonzero(self.y_adapt == label)
            if len(candidates) < k:
                raise ValueError(f"Only {len(candidates)} target samples available for class {label}; requested {k}")
            rng = np.random.default_rng(self.seed + 1000 + label)
            selected.extend(rng.permutation(candidates)[:k].tolist())
        selected = np.asarray(selected, dtype=int)
        return self.x_adapt.iloc[selected], self.y_adapt[selected], self.adapt_indices[selected]


def prepare_fewshot(data_dir: Path, mapping_path: Path, source: str,
                    group: str, seed: int, holdout_fraction: float = .40) -> FewshotData:
    if source not in {"2017", "2018"} or not 0 < holdout_fraction < 1:
        raise ValueError("source must be 2017/2018 and holdout_fraction must lie in (0,1)")
    mapping = load_mapping(mapping_path)
    xs, ys = load_features(data_dir / DATA_FILES[source], mapping, group, source)
    xt, yt = load_features(data_dir / DATA_FILES["iot"], mapping, group, "iot")
    tr, _, _ = split_source(ys, seed)
    adapt, hold = train_test_split(np.arange(len(yt)), test_size=holdout_fraction,
                                    stratify=yt, random_state=seed)
    return FewshotData(xs.iloc[tr], ys[tr], xt.iloc[adapt], yt[adapt],
                       xt.iloc[hold], yt[hold], tr, adapt, hold, seed)


def _matrix(frame: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return frame.fillna(medians).to_numpy(np.float32)


def fit_fewshot(data: FewshotData, k: int, method: str, alpha: float,
                model_type: str, jobs: int):
    if method not in METHODS or not 0 <= alpha <= 1:
        raise ValueError("Unknown method or alpha outside [0,1]")
    if method != "source_only" and k == 0:
        raise ValueError("Target-adaptation methods require k > 0")
    if method == "source_only" and k != 0:
        raise ValueError("source_only requires k=0")
    if method == "weighted" and not 0 < alpha < 1:
        raise ValueError("weighted requires alpha within (0,1)")
    xt, yt, picked = data.target_sample(k)
    if method == "source_only":
        train_x, train_y = data.x_source, data.y_source
        medians = train_x.median().fillna(0)
        weights = compute_sample_weight(class_weight="balanced", y=train_y)
    elif method == "target_only":
        train_x, train_y = xt, yt
        medians = train_x.median().fillna(0)
        weights = compute_sample_weight(class_weight="balanced", y=train_y)
    else:
        train_x = pd.concat([data.x_source, xt], ignore_index=True)
        train_y = np.concatenate([data.y_source, yt])
        medians = data.x_source.median().fillna(0)
        if method == "pooled":
            weights = compute_sample_weight(class_weight="balanced", y=train_y)
        else:
            source_w = compute_sample_weight(class_weight="balanced", y=data.y_source)
            target_w = compute_sample_weight(class_weight="balanced", y=yt)
            # Class balance within each domain; alpha controls total target weight.
            source_w = source_w / source_w.sum() * (1 - alpha)
            target_w = target_w / target_w.sum() * alpha
            weights = np.concatenate([source_w, target_w]) * len(train_y)
    model = make_model(model_type, data.seed, jobs)
    x = _matrix(train_x, medians)
    if model_type == "logistic_regression":
        model.fit(x, train_y, logisticregression__sample_weight=weights)
    else:
        model.fit(x, train_y, sample_weight=weights)
    return model, medians, picked


def evaluate_fewshot(data: FewshotData, k: int, method: str, alpha: float,
                     model_type: str, jobs: int, *, drifts: list[float] | None = None):
    model, medians, picked = fit_fewshot(data, k, method, alpha, model_type, jobs)
    cases = [(0.0, "none", data.x_holdout)]
    for eps in drifts or []:
        if eps <= 0 or eps > 1:
            raise ValueError("Drift epsilons must lie in (0, 1]")
        for direction in ("up", "down"):
            cases.append((eps, direction, perturb_features(data.x_holdout, eps, direction)))
    rows = []
    for eps, direction, frame in cases:
        scores = model.predict_proba(_matrix(frame, medians))[:, 1]
        # Fixed threshold: no labels beyond the k-per-class adaptation budget tune it.
        m = metrics(data.y_holdout, scores, .5)
        rows.append({"seed": data.seed, "method": method, "model": model_type,
                     "k_per_class": k, "target_labels_used": int(2 * k if method != "source_only" else 0),
                     "alpha": alpha if method == "weighted" else None,
                     "epsilon": float(eps), "direction": direction,
                     "n_holdout": len(data.y_holdout), "n_target_adapt_pool": len(data.y_adapt),
                     "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                     "auroc": m["auroc"], "auprc": m["auprc"],
                     "fpr": m["fpr"], "fnr": m["fnr"],
                     "total_cost": m["total_cost"], "cost_per_sample": m["cost_per_sample"],
                     "fp_cost": m["fp_cost"], "fn_cost": m["fn_cost"],
                     "attack_prevalence": m["attack_prevalence"],
                     "tn": m["confusion"]["tn"], "fp": m["confusion"]["fp"],
                     "fn": m["confusion"]["fn"], "tp": m["confusion"]["tp"],
                     "picked_target_indices": ";".join(map(str, picked)) if method != "source_only" else ""})
    return rows
