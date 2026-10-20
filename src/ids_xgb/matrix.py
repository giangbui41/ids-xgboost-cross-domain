"""E1--E4 experiment matrix. All target results here are developmental."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .data import impute_from_train, load_features, load_mapping, overlap_report, sha256
from .experiment import MODEL_TYPES, make_model, metrics, save_json, select_threshold

DATA_FILES = {
    "2017": "cicids2017_balanced_77feats.csv",
    "2018": "cleaned_data_sampled.csv",
    "iot": "ciciot2023_dev.csv",
    "iot_final": "ciciot2023_final_test.csv",
}
SCENARIOS = {
    "E1_2017": ("in_domain", "2017", "2017"),
    "E1_2018": ("in_domain", "2018", "2018"),
    "E1_IoT": ("in_domain", "iot", "iot"),
    "E2_17to18": ("related", "2017", "2018"),
    "E2_18to17": ("related", "2018", "2017"),
    "E3_17toIoT": ("heterogeneous", "2017", "iot"),
    "E3_18toIoT": ("heterogeneous", "2018", "iot"),
    "E4_17plus18toIoT": ("multisource", "2017+2018", "iot"),
}


def native_frame(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, low_memory=False)
    if "Label" not in frame:
        raise ValueError(f"Missing Label in {path}")
    label = pd.to_numeric(frame.pop("Label"), errors="raise")
    if not np.isfinite(label.to_numpy()).all() or (label < 0).any():
        raise ValueError(f"Invalid Label in {path}")
    y = label.ne(0).to_numpy(np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError(f"Both classes required: {path}")
    excluded = re.compile(r"(^|_)(flow_?id|timestamp|src_?ip|dst_?ip|source_?ip|destination_?ip)($|_)")
    keep = [c for c in frame if not excluded.search(re.sub(r"[^a-z0-9]+", "_", c.lower()))]
    x = frame[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x, y


def common_related(source: pd.DataFrame, target: pd.DataFrame,
                   mapping: dict | None = None, source_domain: str = "2017",
                   target_domain: str = "2018"):
    def normalize(name):
        return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")

    # Explicit aliases take precedence over exact normalized names. Never guess
    # equivalence from a partial string match (e.g. mean versus minimum).
    aliases = {}
    if mapping is not None:
        for domain in (source_domain, target_domain):
            aliases[domain] = {normalize(spec[domain]): canonical
                               for canonical, spec in mapping["features"].items()}

    def index(frame, domain):
        names = {}
        for col in frame:
            normalized = normalize(col)
            name = aliases.get(domain, {}).get(normalized, normalized)
            if name in names:
                raise ValueError(f"Ambiguous normalized column: {name}")
            names[name] = col
        return names
    a, b = index(source, source_domain), index(target, target_domain)
    names = sorted(set(a) & set(b))
    if len(names) < 2:
        raise ValueError("Insufficient shared features after explicit mapping and name normalization")
    return source[[a[n] for n in names]].set_axis(names, axis=1), target[[b[n] for n in names]].set_axis(names, axis=1), names


def split_source(y: np.ndarray, seed: int):
    tv, test = train_test_split(np.arange(len(y)), test_size=.20, stratify=y, random_state=seed)
    tr, val = train_test_split(tv, test_size=.125, stratify=y[tv], random_state=seed)
    return tr, val, test


def fit_model(model_type, seed, jobs, xtrain, ytrain, weights, model_params=None):
    model = make_model(model_type, seed, jobs)
    if model_params:
        model.set_params(**model_params)
    if model_type == "logistic_regression":
        model.fit(xtrain, ytrain, logisticregression__sample_weight=weights)
    else:
        model.fit(xtrain, ytrain, sample_weight=weights)
    return model


def run_one(name: str, model_type: str, data_dir: Path, mapping_path: Path,
            group: str, seed: int, jobs: int, model_params: dict | None = None) -> dict:
    if name not in SCENARIOS or model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown scenario/model: {name}/{model_type}")
    mode, source, target = SCENARIOS[name]
    mapping = load_mapping(mapping_path)
    paths = {domain: data_dir / file for domain, file in DATA_FILES.items()}
    input_hashes = {}
    alignment = None
    split_audit = None
    if mode == "in_domain":
        # Native in-domain baseline: all numeric fields, not the seven shared features.
        x, y = native_frame(paths[source]); tr, va, te = split_source(y, seed)
        split_audit = {"train_validation": overlap_report(x.iloc[tr], x.iloc[va]),
                       "train_test": overlap_report(x.iloc[tr], x.iloc[te])}
        a, (b, c), _ = impute_from_train(x.iloc[tr], x.iloc[va], x.iloc[te])
        ytr, yva, yte = y[tr], y[va], y[te]
        feature_names = list(x)
        input_hashes[source] = sha256(paths[source])
    elif mode == "related":
        xs, ys = native_frame(paths[source]); xt, yt = native_frame(paths[target])
        original_source, original_target = list(xs), list(xt)
        xs, xt, feature_names = common_related(xs, xt, mapping, source, target)
        alignment = {"source_columns": original_source, "target_columns": original_target,
                     "selected_features": feature_names,
                     "source_columns_excluded_count": len(original_source) - len(feature_names),
                     "target_columns_excluded_count": len(original_target) - len(feature_names),
                     "rule": "Explicit feature mapping, then exact normalized names"}
        tr, va, _ = split_source(ys, seed)
        a, (b, c), _ = impute_from_train(xs.iloc[tr], xs.iloc[va], xt)
        ytr, yva, yte = ys[tr], ys[va], yt
        input_hashes = {k: sha256(paths[k]) for k in (source, target)}
    elif mode == "heterogeneous":
        xs, ys = load_features(paths[source], mapping, group, source)
        xt, yt = load_features(paths["iot"], mapping, group, "iot")
        tr, va, _ = split_source(ys, seed)
        a, (b, c), _ = impute_from_train(xs.iloc[tr], xs.iloc[va], xt)
        ytr, yva, yte = ys[tr], ys[va], yt
        feature_names = list(xs)
        input_hashes = {k: sha256(paths[k]) for k in (source, "iot")}
    else:
        xs17, y17 = load_features(paths["2017"], mapping, group, "2017")
        xs18, y18 = load_features(paths["2018"], mapping, group, "2018")
        xt, yt = load_features(paths["iot"], mapping, group, "iot")
        t17, v17, _ = split_source(y17, seed)
        t18, v18, _ = split_source(y18, seed)
        train_frame = pd.concat([xs17.iloc[t17], xs18.iloc[t18]], ignore_index=True)
        val_frame = pd.concat([xs17.iloc[v17], xs18.iloc[v18]], ignore_index=True)
        a, (b, c), _ = impute_from_train(train_frame, val_frame, xt)
        ytr = np.concatenate([y17[t17], y18[t18]])
        yva = np.concatenate([y17[v17], y18[v18]])
        yte = yt
        feature_names = list(xs17)
        input_hashes = {k: sha256(paths[k]) for k in ("2017", "2018", "iot")}

    weights = compute_sample_weight(class_weight="balanced", y=ytr)
    if mode == "multisource":
        # Equal total weight for each source, then normalize mean weight to one.
        n17, n18 = len(t17), len(t18)
        weights[:n17] *= .5 / weights[:n17].sum()
        weights[n17:] *= .5 / weights[n17:].sum()
        weights *= len(weights)
    model = fit_model(model_type, seed, jobs, a, ytr, weights, model_params)
    score_val = model.predict_proba(b)[:, 1]
    threshold = select_threshold(yva, score_val)
    margin = float(np.quantile(np.abs(score_val - threshold), .10))
    result = {
        "scenario": name, "mode": mode, "model": model_type,
        "model_params_override": model_params or {},
        "source": source, "target": target, "seed": seed, "group": "native" if mode == "in_domain" else ("mapped_common" if mode == "related" else group),
        "feature_alignment": alignment,
        "split_audit": split_audit,
        "n_features": len(feature_names), "feature_names": feature_names,
        "n_train": len(ytr), "n_validation": len(yva), "n_evaluation": len(yte),
        "input_sha256": input_hashes, "mapping_sha256": sha256(mapping_path),
        "source_validation": metrics(yva, score_val, threshold, margin),
        "evaluation": metrics(yte, model.predict_proba(c)[:, 1], threshold, margin),
        "evaluation_role": "internal_source_test" if mode == "in_domain" else "cross_domain_diagnostic",
        "target_labels_in_training_or_threshold": mode == "in_domain",
        "note": "These experiments do not replicate old E1-E4 architectures, splits, or target adaptation. IoT FINAL is never read.",
    }
    return result


def run_matrix(scenarios: list[str], models: list[str], data_dir: Path,
               mapping_path: Path, group: str, seed: int, jobs: int, out: Path) -> list[dict]:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = []
    for scenario in scenarios:
        for model in models:
            item = run_one(scenario, model, data_dir, mapping_path, group, seed, jobs)
            save_json(out / f"{scenario}_{model}_s{seed}.json", item)
            metric = item["evaluation"]
            rows.append({"scenario": scenario, "model": model, "seed": seed,
                         "group": item["group"], "n_features": item["n_features"],
                         "macro_f1": metric["macro_f1"], "balanced_accuracy": metric["balanced_accuracy"],
                         "auroc": metric["auroc"], "auprc": metric["auprc"],
                         "fpr": metric["fpr"], "fnr": metric["fnr"],
                         "attack_prevalence": metric["attack_prevalence"]})
            pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
            print(f"{scenario} {model}: macro_f1={metric['macro_f1']:.4f} auroc={metric['auroc']:.4f}", flush=True)
    return rows


def run_sensitivity(data_dir: Path, mapping_path: Path, group: str, seed: int,
                    jobs: int, depths: list[int], out: Path) -> list[dict]:
    """E5 analogue: source-validated XGBoost tree depth; never DANN lambda."""
    if out.exists():
        raise FileExistsError(out)
    if not depths or any(d < 1 or d > 16 for d in depths) or len(set(depths)) != len(depths):
        raise ValueError("Depths must be distinct integers from 1 to 16")
    out.mkdir(parents=True)
    rows = []
    for depth in depths:
        item = run_one("E3_18toIoT", "xgboost", data_dir, mapping_path,
                       group, seed, jobs, {"max_depth": depth})
        save_json(out / f"E5_depth_{depth}_s{seed}.json", item)
        m = item["evaluation"]
        rows.append({"depth": depth,
                     "source_validation_macro_f1": item["source_validation"]["macro_f1"],
                     "iot_dev_macro_f1": m["macro_f1"], "iot_dev_auroc": m["auroc"],
                     "iot_dev_fpr": m["fpr"], "iot_dev_fnr": m["fnr"]})
        pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)
    return rows
