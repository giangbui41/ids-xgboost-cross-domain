"""E8: controlled, cost-sensitive extensions using IoT DEV only.

The 60% adaptation pool supplies budgeted labels and unlabeled examples. The
40% evaluation pool is never passed to fitting, except in explicitly labelled
transductive TTA variants. FINAL is outside this runner's input contract.
"""
from __future__ import annotations

import json
import math
import re
import joblib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .data import load_features, load_mapping, overlap_report
from .experiment import make_model, metrics, select_cost_threshold
from .fewshot import prepare_fewshot
from .matrix import DATA_FILES, split_source

METHODS = {"source_only", "target_only", "pooled", "weighted", "adaptive", "weighted_adaptive", "self_training", "hybrid"}
XGB_PARAMS = {"n_estimators", "max_depth", "learning_rate", "min_child_weight", "reg_lambda",
              "reg_alpha", "gamma", "subsample", "colsample_bytree", "max_bin"}


def _positive_int(value):
    return type(value) is int and value > 0


def validate_improvements(config: dict) -> None:
    """Reject invalid experiments before hashing data or creating run directories."""
    allowed = {"stage", "description", "sources", "group", "seeds", "jobs", "holdout_fraction",
               "k_per_class", "alpha", "costs", "threshold", "model_params", "variants",
               "adaptive", "self_training", "hybrid", "explanations", "max_source_samples",
               "max_unlabeled_samples", "selection_metric", "calibration"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown E8 options: {sorted(unknown)}")
    if config.get("selection_metric", "total_cost") != "total_cost":
        raise ValueError("E8 selection_metric must be total_cost (lower is better)")
    for key in ("max_source_samples", "max_unlabeled_samples"):
        if config.get(key) is not None and not _positive_int(config[key]):
            raise ValueError(f"{key} must be null or a positive integer")
    if config.get("max_source_samples") is not None and config["max_source_samples"] < 2:
        raise ValueError("max_source_samples must retain both source classes")
    costs = config.get("costs", {})
    if set(costs) - {"fp", "fn"}:
        raise ValueError("costs supports only fp and fn")
    a, b = costs.get("fp", 100), costs.get("fn", 1)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (a, b)) or a + b <= 0:
        raise ValueError("Costs must be finite, nonnegative, and not both zero")
    threshold = config.get("threshold", {})
    if set(threshold) - {"strategy", "value", "calibration_fraction"}:
        raise ValueError("Unknown threshold option")
    if threshold.get("strategy", "fixed") not in {"fixed", "budget_validation_cost"}:
        raise ValueError("Unknown threshold strategy")
    if not isinstance(threshold.get("value", .5), (int, float)) or not 0 <= threshold.get("value", .5) <= 1:
        raise ValueError("threshold value must lie in [0,1]")
    if not 0 < threshold.get("calibration_fraction", .3) < 1:
        raise ValueError("calibration_fraction must lie in (0,1)")
    if threshold.get("strategy") == "budget_validation_cost":
        for k in config.get("k_per_class", []):
            if _positive_int(k) and int(np.ceil(k * threshold.get("calibration_fraction", .3))) >= k:
                raise ValueError("K too small for separate training and calibration labels")
    params = config.get("model_params", {})
    if not isinstance(params, dict) or set(params) - XGB_PARAMS:
        raise ValueError("Unsupported XGBoost model_params")
    for key, value in params.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid model parameter: {key}")
        if key in {"n_estimators", "max_depth", "max_bin"} and not _positive_int(value):
            raise ValueError(f"{key} must be a positive integer")
        if value < 0 or (key in {"learning_rate", "subsample", "colsample_bytree"} and not 0 < value <= 1):
            raise ValueError(f"Invalid model parameter: {key}")
    variants = config.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("E8 requires variants")
    names = set()
    for v in variants:
        if not isinstance(v, dict) or set(v) - {"name", "method", "feature_engineering", "tta", "adaptive_weighting", "calibrate"}:
            raise ValueError("Invalid variant options")
        name = v.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("Variant names must be unique safe identifiers")
        names.add(name)
        if v.get("method") not in METHODS:
            raise ValueError("Unknown E8 method")
        for flag in ("feature_engineering", "adaptive_weighting", "calibrate"):
            if flag in v and type(v[flag]) is not bool:
                raise ValueError(f"{flag} must be boolean")
        tta = v.get("tta", "none")
        if tta not in {"none", "entropy", "self_training"}:
            raise ValueError("tta must be none, entropy, or self_training")
        if v.get("calibrate", False) and (tta != "none" or v["method"] == "hybrid"):
            raise ValueError("Use separate calibration and TTA/hybrid variants")
        if v["method"] == "hybrid" and tta == "self_training":
            raise ValueError("Hybrid supports entropy TTA only; use XGBoost for TTA self-training")
        if v["method"] in {"source_only", "target_only"} and v.get("adaptive_weighting", False):
            raise ValueError("Adaptive weighting requires a mixed-domain method")
        if v["method"] == "pooled" and v.get("adaptive_weighting", False):
            raise ValueError("Use method=adaptive for source instance weighting, not pooled")
    specifications = {
        "adaptive": {"strategy", "min_weight", "max_weight"},
        "self_training": {"iterations", "confidence", "pseudo_weight", "max_per_class"},
        "hybrid": {"hidden_dim", "embedding_dim", "epochs", "batch_size", "learning_rate", "lambda_domain", "max_unlabeled"},
        "explanations": {"enabled", "max_samples"},
        "calibration": {"method", "fraction", "n_bins", "tolerance"},
    }
    for section, keys in specifications.items():
        options = config.get(section, {})
        if not isinstance(options, dict) or set(options) - keys:
            raise ValueError(f"Unknown {section} options")
    ssl = config.get("self_training", {})
    cal = config.get("calibration", {})
    if (cal.get("method", "sigmoid") not in {"sigmoid", "isotonic"}
            or not 0 < cal.get("fraction", .3) < 1
            or not _positive_int(cal.get("n_bins", 10)) or cal.get("n_bins", 10) < 2
            or not 0 <= cal.get("tolerance", .05) <= 1):
        raise ValueError("Invalid calibration options")
    if any(v.get("calibrate", False) and v["method"] != "source_only" for v in variants):
        fraction = (threshold.get("calibration_fraction", .3)
                    if threshold.get("strategy") == "budget_validation_cost" else cal.get("fraction", .3))
        if any(_positive_int(k) and int(np.ceil(k * fraction)) >= k for k in config.get("k_per_class", [])):
            raise ValueError("K too small for separate training and calibration labels")
    for key in ("iterations", "max_per_class"):
        if key in ssl and not _positive_int(ssl[key]):
            raise ValueError(f"self_training.{key} must be positive")
    if not .5 < ssl.get("confidence", .95) <= 1 or not 0 < ssl.get("pseudo_weight", .2) <= 1:
        raise ValueError("Invalid self-training confidence/pseudo_weight")
    adaptive = config.get("adaptive", {})
    if adaptive.get("strategy", "density") not in {"density", "uncertainty"}:
        raise ValueError("Unknown adaptive weighting strategy")
    if not 0 < adaptive.get("min_weight", .1) <= 1 <= adaptive.get("max_weight", 10):
        raise ValueError("Adaptive bounds must include one")
    exp = config.get("explanations", {})
    if type(exp.get("enabled", True)) is not bool or not _positive_int(exp.get("max_samples", 256)):
        raise ValueError("Invalid explanations options")
    h = config.get("hybrid", {})
    for key in ("hidden_dim", "embedding_dim", "epochs", "batch_size", "max_unlabeled"):
        if key in h and not _positive_int(h[key]):
            raise ValueError(f"hybrid.{key} must be positive")
    for key in ("learning_rate", "lambda_domain"):
        if key in h and (type(h[key]) not in (int, float) or not math.isfinite(h[key]) or h[key] < 0):
            raise ValueError(f"Invalid hybrid.{key}")
    if h.get("learning_rate", .001) == 0:
        raise ValueError("Hybrid learning_rate must be positive")


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _weights(ys, yt, method, alpha, multiplier=None):
    if method == "source_only":
        return compute_sample_weight("balanced", ys)
    if method == "target_only":
        return compute_sample_weight("balanced", yt)
    if method == "pooled":
        return compute_sample_weight("balanced", np.r_[ys, yt])
    sw = compute_sample_weight("balanced", ys)
    if multiplier is not None:
        sw *= multiplier
    tw = compute_sample_weight("balanced", yt)
    return np.r_[sw / sw.sum() * (1 - alpha), tw / tw.sum() * alpha] * (len(ys) + len(yt))


def _scores(model, x):
    # Make threshold=1 a well-defined all-benign policy, even with saturated trees.
    return np.clip(model.predict_proba(x)[:, 1].astype(float), 0, np.nextafter(1.0, 0.0))


def _source_subset(y, cap, seed):
    """Retain both classes without requiring a stratified unused complement."""
    if cap is None or cap >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    zero, one = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    nzero = min(len(zero), max(1, int(round(cap * len(zero) / len(y)))))
    nzero = max(nzero, cap - len(one))
    nzero = min(nzero, cap - 1)
    return np.sort(np.r_[rng.choice(zero, nzero, replace=False), rng.choice(one, cap - nzero, replace=False)])


def run_improvements(config: dict, data_dir: Path, mapping_path: Path, work: Path) -> pd.DataFrame:
    from .adaptation import adaptive_source_weights, entropy_temperature, self_train
    from .calibration import calibrate_prefit, calibration_metrics
    from .explanations import export_tree_explanations
    from .features import FeatureEngineer

    rows = []
    stage = config.get("stage", "E8")
    cal_options = config.get("calibration", {})
    cal_metrics_options = {"n_bins": cal_options.get("n_bins", 10), "tolerance": cal_options.get("tolerance", .05)}
    costs = config.get("costs", {})
    cost_args = {"fp_cost": costs.get("fp", 100), "fn_cost": costs.get("fn", 1)}
    ssl_options = config.get("self_training", {})
    for seed in config["seeds"]:
        for source in config["sources"]:
            data = prepare_fewshot(data_dir, mapping_path, source, config.get("group", "seven"),
                                   seed, config.get("holdout_fraction", .4))
            data.target_sample(max(config["k_per_class"]))
            source_idx = _source_subset(data.y_source, config.get("max_source_samples"), seed)
            source_frame = data.x_source.iloc[source_idx]
            ys = data.y_source[source_idx]
            _write(work / f"split_{source}_s{seed}.json", {
                "source_train_indices": data.source_train_indices[source_idx].tolist(),
                "adapt_indices": data.adapt_indices.tolist(), "holdout_indices": data.holdout_indices.tolist(),
                "overlap": overlap_report(data.x_adapt, data.x_holdout)})
            source_calibration = None
            for variant in config["variants"]:
                method = variant["method"]
                for k in ([0] if method == "source_only" else config["k_per_class"]):
                    out = work / f"{variant['name']}_{source}_s{seed}_k{k}"
                    out.mkdir()
                    target_frame, target_y, picked = data.target_sample(k)
                    target_train_idx = np.arange(len(target_y))
                    calibration_idx = np.array([], dtype=int)
                    policy = config.get("threshold", {}).get("strategy", "fixed")
                    calibrate = variant.get("calibrate", False)
                    if (policy == "budget_validation_cost" or calibrate) and k:
                        fraction = (config.get("threshold", {}).get("calibration_fraction", .3)
                                    if policy == "budget_validation_cost" else cal_options.get("fraction", .3))
                        ncal = max(1, int(np.ceil(k * fraction)))
                        if ncal >= k:
                            raise ValueError("K too small to reserve calibration labels and retain both training classes")
                        # The selected K-per-class budget includes calibration labels.
                        calibration_idx = np.r_[np.arange(ncal), np.arange(k, k + ncal)]
                        target_train_idx = np.setdiff1d(target_train_idx, calibration_idx)
                    xt_frame = target_frame.iloc[target_train_idx]
                    yt = target_y[target_train_idx]
                    unlabeled_idx = np.flatnonzero(~np.isin(data.adapt_indices, picked))
                    cap = config.get("max_unlabeled_samples")
                    if cap and len(unlabeled_idx) > cap:
                        unlabeled_idx = np.sort(np.random.default_rng(seed).choice(unlabeled_idx, cap, replace=False))
                    xu_frame = data.x_adapt.iloc[unlabeled_idx]
                    engineer = FeatureEngineer(enabled=variant.get("feature_engineering", True))
                    fit_frame = xt_frame if method == "target_only" else source_frame
                    engineer.fit(fit_frame)
                    medians = engineer.transform(fit_frame).median().fillna(0)

                    def matrix(frame):
                        transformed = engineer.transform(frame).replace([np.inf, -np.inf], np.nan).fillna(medians)
                        # Keep interactions representable as float32 after arithmetic.
                        return transformed.clip(-1e30, 1e30).to_numpy(np.float32)

                    xs, xt, xu, xh = (matrix(f) for f in (source_frame, xt_frame, xu_frame, data.x_holdout))
                    names = list(medians.index)
                    xtrain = xs if method == "source_only" else xt if method == "target_only" else np.concatenate([xs, xt])
                    ytrain = ys if method == "source_only" else yt if method == "target_only" else np.r_[ys, yt]
                    diagnostics = {}
                    multiplier = None
                    start = perf_counter()
                    if method in {"adaptive", "weighted_adaptive"} or variant.get("adaptive_weighting", False):
                        multiplier, diagnostics = adaptive_source_weights(xs, ys, xt, yt, xu,
                                                                          seed=seed, **config.get("adaptive", {}))
                    weights = _weights(ys, yt, method, config.get("alpha", .5), multiplier)

                    def factory():
                        model = make_model("xgboost", seed, config.get("jobs", 4))
                        model.set_params(**config.get("model_params", {}))
                        return model

                    history = []
                    if method == "hybrid":
                        from .hybrid import HybridDANNXGB
                        model = HybridDANNXGB(seed=seed, jobs=config.get("jobs", 4),
                                             xgb_params=config.get("model_params", {}), **config.get("hybrid", {}))
                        model.fit(xs, ys, xt, yt, xu, sample_weight=weights)
                    elif method == "self_training":
                        model, history = self_train(factory, xtrain, ytrain, weights, xu, seed=seed, **ssl_options)
                    else:
                        model = factory()
                        model.fit(xtrain, ytrain, sample_weight=weights)
                    train_seconds = perf_counter() - start
                    base_model = model
                    threshold = config.get("threshold", {}).get("value", .5)
                    calibration_info = {}
                    if policy == "budget_validation_cost" or calibrate:
                        if method == "source_only":
                            if source_calibration is None:
                                all_x, all_y = load_features(data_dir / DATA_FILES[source], load_mapping(mapping_path),
                                                            config.get("group", "seven"), source)
                                _, va, _ = split_source(all_y, seed)
                                source_calibration = all_x.iloc[va], all_y[va]
                            xc, yc = matrix(source_calibration[0]), source_calibration[1]
                        else:
                            xc, yc = matrix(target_frame.iloc[calibration_idx]), target_y[calibration_idx]
                        if calibrate:
                            start = perf_counter()
                            model = calibrate_prefit(base_model, xc, yc, method=cal_options.get("method", "sigmoid"))
                            train_seconds += perf_counter() - start
                            calibration_info = {"method": cal_options.get("method", "sigmoid"),
                                                "fit_domain": "source_validation" if method == "source_only" else "budgeted_target",
                                                "fit_rows": len(yc), "target_budget_includes_calibration": True}
                    if policy == "budget_validation_cost":
                        threshold = select_cost_threshold(yc, _scores(model, xc), **cost_args)
                    start = perf_counter()
                    original_scores = _scores(model, xh)
                    inference_seconds = perf_counter() - start
                    scores = original_scores.copy()
                    tta = variant.get("tta", "none")
                    tta_info = {}
                    tta_seconds = 0.0
                    if tta != "none":
                        start = perf_counter()
                        if tta == "entropy":
                            scores, tta_info = entropy_temperature(scores)
                            scores = np.minimum(scores, np.nextafter(1.0, 0.0))
                        else:
                            # Features only. The holdout labels enter exclusively below, in metrics().
                            if method == "hybrid":
                                raise ValueError("Hybrid TTA self-training is unsupported")
                            pool = np.concatenate([xu, xh]) if method == "self_training" else xh
                            model, tta_history = self_train(factory, xtrain, ytrain, weights, pool,
                                                           seed=seed, **ssl_options)
                            scores = _scores(model, xh)
                            tta_info = {"history": tta_history, "pool_rows": len(pool),
                                        "mode": "joint_transductive_retraining" if method == "self_training" else "transductive_retraining"}
                        tta_seconds = perf_counter() - start
                    result = metrics(data.y_holdout, scores, threshold, **cost_args)
                    before = metrics(data.y_holdout, original_scores, threshold, **cost_args)
                    probability_metrics = calibration_metrics(data.y_holdout, scores, **cal_metrics_options)
                    raw_scores = _scores(base_model, xh) if calibrate else original_scores
                    raw_probability_metrics = calibration_metrics(data.y_holdout, raw_scores, **cal_metrics_options)
                    if calibrate:
                        _write(out / "calibration.json", {**calibration_info,
                               "evaluation_role": "holdout_diagnostics_only_not_used_to_fit_or_select",
                               "before": raw_probability_metrics, "after": probability_metrics})
                    metric_names = ["macro_f1", "balanced_accuracy", "auroc", "auprc", "fpr", "fnr", "total_cost",
                                    "cost_per_sample", "always_benign_cost", "always_attack_cost", "attack_prevalence"]
                    row = {"stage": stage, "source": source, "variant": variant["name"], "method": method,
                           "model": "dann_xgboost" if method == "hybrid" else "xgboost", "seed": seed,
                           "k_per_class": k, "feature_engineering": variant.get("feature_engineering", True),
                           "n_features": len(names), "alpha": config.get("alpha", .5),
                           "tta": tta, "evaluation_role": "transductive_holdout" if tta != "none" else "independent_holdout",
                           "threshold_strategy": policy, "threshold": threshold, **cost_args,
                           "calibration": cal_options.get("method", "sigmoid") if calibrate else "none",
                           "ece": probability_metrics["ece"], "brier_score": probability_metrics["brier_score"],
                           "log_loss": probability_metrics["log_loss"], "before_calibration_ece": raw_probability_metrics["ece"],
                           "before_calibration_brier_score": raw_probability_metrics["brier_score"],
                           "target_labels_used": len(picked), "target_training_labels": len(yt),
                           "target_calibration_labels": len(calibration_idx), "n_source": len(ys) if method != "target_only" else 0,
                           "n_unlabeled_available": len(xu), "n_holdout": len(xh),
                           "train_seconds": train_seconds, "inference_seconds": inference_seconds, "tta_seconds": tta_seconds,
                           "before_tta_total_cost": before["total_cost"], "before_tta_macro_f1": before["macro_f1"],
                           **{name: result[name] for name in metric_names}, **result["confusion"]}
                    rows.append(row)
                    np.savez_compressed(out / "predictions.npz", row_id=data.holdout_indices, y=data.y_holdout,
                                        score=scores, before_tta_score=original_scores, prediction=(scores >= threshold).astype(np.int8))
                    _write(out / "run.json", {**row, "feature_names": names, "medians": medians.to_dict(),
                           "model_params": config.get("model_params", {}), "selected_target_indices": picked.tolist(),
                           "calibration_target_indices": picked[calibration_idx].tolist(),
                           "unlabeled_target_indices": data.adapt_indices[unlabeled_idx].tolist(),
                           "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                           "source_weight_sum": float(weights[:len(ys)].sum()) if method != "target_only" else 0.0,
                           "target_weight_sum": float(weights.sum()) if method == "target_only" else float(weights[len(ys):].sum()),
                           "adaptive_diagnostics": diagnostics, "self_training_history": history, "tta_diagnostics": tta_info,
                           "calibration_info": calibration_info,
                           "note": "Cost assumptions are user-defined. DEV row split is not verified device/time independence. No FINAL data used."})
                    if method == "hybrid":
                        model.save(out / "hybrid_model")
                    elif calibrate:
                        base_model.save_model(out / "model.json")
                        joblib.dump(model, out / "calibrated_model.joblib")
                    else:
                        model.save_model(out / "model.json")
                    explanations = config.get("explanations", {})
                    if explanations.get("enabled", True):
                        if method == "hybrid":
                            explain_model = model.classifier_
                            # Sample before encoder inference to keep explanation memory bounded.
                            n = explanations.get("max_samples", 256)
                            srng, trng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
                            source_take = srng.choice(len(xs), min(n, len(xs)), replace=False)
                            target_take = trng.choice(len(xh), min(n, len(xh)), replace=False)
                            sx, hx = model.transform(xs[source_take]), model.transform(xh[target_take])
                            explain_names = model.embedding_feature_names_
                        else:
                            explain_model, sx, hx, explain_names = base_model if calibrate else model, xs, xh, names
                        report = export_tree_explanations(explain_model, sx, hx, explain_names, out / "explanations",
                                                          seed=seed, max_samples=explanations.get("max_samples", 256))
                        source_positions = np.asarray(report["source_row_positions"], dtype=int)
                        target_positions = np.asarray(report["target_row_positions"], dtype=int)
                        if method == "hybrid":
                            source_positions, target_positions = source_take[source_positions], target_take[target_positions]
                        _write(out / "explanations" / "dataset_row_ids.json", {
                            "source_csv_row_ids": data.source_train_indices[source_idx][source_positions].tolist(),
                            "target_dev_row_ids": data.holdout_indices[target_positions].tolist()})
                        _write(out / "explanations" / "decision_policy.json", {
                            "threshold": threshold, "tta": tta, "calibration": row["calibration"],
                            "note": "TreeSHAP explains the base tree raw margin, before probability calibration or entropy temperature. For calibrated predictions load calibrated_model.joblib. Contributions are associations, not proof of causal bias."})
                    print(f"{stage} {variant['name']} {source} seed={seed} K={k}: cost={row['total_cost']:.1f}, F1={row['macro_f1']:.4f}", flush=True)
    return pd.DataFrame(rows)
