"""Numbered E0--E8 pipeline with versioned CSV snapshots and rollback."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .cli import audit
from .data import claim_final_evaluation, sha256
from .fewshot import METHODS, evaluate_fewshot, prepare_fewshot
from .fewshot import fit_fewshot
from .data import load_features, load_mapping, overlap_report
from .experiment import MODEL_TYPES, metrics
from .matrix import DATA_FILES, SCENARIOS, run_one

ROOT = Path(__file__).resolve().parents[2]
STAGE_CONFIGS = {
    "E0": "E0_data_audit.json",
    "E1": "E1_in_domain.json",
    "E2": "E2_related_transfer.json",
    "E3": "E3_zero_shot_iot.json",
    "E4": "E4_fewshot.json",
    "E5": "E5_alpha_sensitivity.json",
    "E6": "E6_measurement_drift.json",
    "E7": "E7_final_evaluation.json",
    "E8": "E8_improvements.json",
}
STAGES = tuple(STAGE_CONFIGS)


def _fraction(value, *, include_one: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and
            value > 0 and (value <= 1 if include_one else value < 1))


def _distinct_list(value) -> bool:
    return (isinstance(value, list) and bool(value) and
            all(isinstance(item, (str, int, float)) for item in value) and
            len(set(value)) == len(value))


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _input_hashes(data_dir: Path, stage: str, mapping: Path, config: dict,
                  *, include_final: bool = True) -> dict:
    if stage == "E0":
        required = {"2017", "2018", "iot"}
    elif stage in {"E1", "E2", "E3"}:
        required = {domain for scenario in config["scenarios"]
                    for domain in SCENARIOS[scenario][1:]}
    elif stage == "E7":
        required = {config["source"], "iot"}
        if include_final:
            required.add("iot_final")
    else:
        required = {*config.get("sources", ["2018"]), "iot"}
    return {**{domain: sha256(data_dir / DATA_FILES[domain]) for domain in required},
            "feature_mapping": sha256(mapping)}


def _code_hashes() -> dict:
    return {p.name: sha256(p) for p in sorted((ROOT / "src" / "ids_xgb").glob("*.py"))}


@contextmanager
def _exclusive_registry(results_dir: Path):
    path = results_dir / ".registry.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("Registry is busy; retry after the other writer finishes") from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def lock_final(config_path: Path, data_dir: Path, mapping: Path, reason: str) -> Path:
    """Record a selection without opening FINAL. Existing locks are immutable."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config("E7", config, mapping)
    if not reason.strip():
        raise ValueError("A selection reason is required")
    lock = {"config_sha256": sha256(config_path), "code_sha256": _code_hashes(),
            "inputs_sha256": _input_hashes(data_dir, "E7", mapping, config, include_final=False),
            "locked_utc": datetime.now(timezone.utc).isoformat(), "reason": reason}
    path = config_path.with_suffix(".lock.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
    return path


def _verify_final_lock(config_path: Path, data_dir: Path, mapping: Path, config: dict) -> dict:
    path = config_path.with_suffix(".lock.json")
    if not path.is_file():
        raise ValueError("E7 requires a lock-final record before FINAL can be read")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("E7 config changed while being read")
    expected = {"config_sha256": sha256(config_path), "code_sha256": _code_hashes(),
                "inputs_sha256": _input_hashes(data_dir, "E7", mapping, config, include_final=False)}
    if any(lock.get(key) != value for key, value in expected.items()):
        raise ValueError("E7 configuration, code, mapping or training data changed since lock-final")
    return lock


def _validate_final_config(config: dict) -> None:
    for key in ("source", "method", "k_per_class", "seed", "model"):
        if config.get(key) in (None, "UNSET"):
            raise ValueError(f"E7 requires a preselected {key} in its locked config")
    if config["source"] not in {"2017", "2018"} or config["method"] not in METHODS:
        raise ValueError("E7 source or method is invalid")
    if config["model"] not in MODEL_TYPES or type(config["seed"]) is not int:
        raise ValueError("E7 model or seed is invalid")
    k = config["k_per_class"]
    if type(k) is not int or k < 0 or (k == 0 and config["method"] != "source_only"):
        raise ValueError("E7 requires nonnegative integer K, positive for few-shot methods")
    if config["method"] == "source_only" and k != 0:
        raise ValueError("E7 source_only requires K=0")
    if config["method"] == "weighted" and not _fraction(config.get("alpha")):
        raise ValueError("E7 weighted alpha must be within (0,1)")


def _validate_config(stage: str, config: dict, mapping: Path) -> None:
    if config.get("stage") != stage:
        raise ValueError("Config stage mismatch")
    if stage == "E7":
        _validate_final_config(config)
    if config.get("group", "seven") not in load_mapping(mapping)["groups"]:
        raise ValueError("Unknown feature group")
    if type(config.get("jobs", 4)) is not int or config.get("jobs", 4) < 1:
        raise ValueError("jobs must be a positive integer")
    if stage == "E0":
        return
    seeds = [config["seed"]] if stage == "E7" else config.get("seeds", [])
    if not _distinct_list(seeds) or any(type(s) is not int or not 0 <= s < 2**32 - 1002 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers below 2**32-1002")
    if stage in {"E1", "E2", "E3"}:
        scenarios, models = config.get("scenarios", []), config.get("models", [])
        if not _distinct_list(scenarios) or any(s not in SCENARIOS or not s.startswith(stage + "_") for s in scenarios):
            raise ValueError("Invalid scenarios for stage")
        if not _distinct_list(models) or any(m not in MODEL_TYPES for m in models):
            raise ValueError("Invalid models")
        return
    if stage == "E8" or (stage == "E4" and "feature_engineering" in config):
        from .improvements import validate_improvements
        if stage == "E4":
            from .fewshot import improvement_config
            config = improvement_config(config)
        sources, budgets = config.get("sources", []), config.get("k_per_class", [])
        if not _distinct_list(sources) or any(s not in {"2017", "2018"} for s in sources):
            raise ValueError("Invalid E8 sources")
        if not _distinct_list(budgets) or any(type(k) is not int or k <= 0 for k in budgets):
            raise ValueError("E8 requires positive integer budgets")
        if not _fraction(config.get("holdout_fraction", .4)) or not _fraction(config.get("alpha", .5)):
            raise ValueError("Invalid E8 holdout_fraction/alpha")
        validate_improvements(config)
        return
    if config.get("model", "xgboost") not in MODEL_TYPES:
        raise ValueError("Unknown model")
    sources = [config["source"]] if stage == "E7" else config.get("sources", ["2018"])
    if not _distinct_list(sources) or any(s not in {"2017", "2018"} for s in sources):
        raise ValueError("Invalid sources")
    if not _fraction(config.get("holdout_fraction", .4)):
        raise ValueError("holdout_fraction must be within (0,1)")
    if stage == "E7":
        return
    budgets = config.get("k_per_class", []) if stage == "E4" else [config.get("k_per_class")]
    if not _distinct_list(budgets) or any(type(k) is not int or k <= 0 for k in budgets):
        raise ValueError("k_per_class must contain positive integer budgets")
    methods = ["weighted"] if stage == "E5" else config.get("methods", [])
    if not _distinct_list(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Invalid methods")
    alphas = config.get("alphas", []) if stage == "E5" else [config.get("alpha", .5)]
    if not _distinct_list(alphas) or any(not _fraction(a) for a in alphas):
        raise ValueError("alpha must be within (0,1)")
    if stage == "E6" and (not _distinct_list(config.get("epsilons")) or
                           any(not _fraction(e, include_one=True) for e in config["epsilons"])):
        raise ValueError("epsilons must be within (0,1]")


def _produce(stage: str, config: dict, data_dir: Path, mapping: Path, work: Path) -> pd.DataFrame:
    rows = []
    if stage == "E0":
        report = audit(mapping, data_dir / DATA_FILES["2017"],
                       data_dir / DATA_FILES["2018"], data_dir / DATA_FILES["iot"],
                       work / "audit.json")
        for domain, details in report["datasets"].items():
            for feature, stats in details["features"].items():
                rows.append({"domain": domain, "feature": feature, "rows": details["rows"],
                             "attack_prevalence": details["attack_prevalence"], **stats})
    elif stage in {"E1", "E2", "E3"}:
        for seed in config["seeds"]:
            for scenario in config["scenarios"]:
                if not scenario.startswith(stage + "_"):
                    raise ValueError(f"Scenario {scenario} does not belong to {stage}")
                for model in config["models"]:
                    item = run_one(scenario, model, data_dir, mapping, config.get("group", "seven"),
                                   seed, config.get("jobs", 4))
                    m = item["evaluation"]
                    rows.append({"stage": stage, "scenario": scenario, "source": item["source"],
                                 "target": item["target"], "model": model, "seed": seed,
                                 "feature_group": item["group"], "n_features": item["n_features"],
                                 "n_evaluation": m["n"], "attack_prevalence": m["attack_prevalence"],
                                 "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                                 "auroc": m["auroc"], "auprc": m["auprc"],
                                 "fpr": m["fpr"], "fnr": m["fnr"],
                                 "total_cost": m["total_cost"], "cost_per_sample": m["cost_per_sample"],
                                 "fp_cost": m["fp_cost"], "fn_cost": m["fn_cost"],
                                 "source_threshold": item["source_validation"]["threshold"],
                                 "evaluation_role": item["evaluation_role"]})
                    (work / f"{scenario}_{model}_s{seed}.json").write_text(
                        json.dumps(item, indent=2) + "\n", encoding="utf-8")
    elif stage == "E4" and "feature_engineering" in config:
        from .fewshot import run_fewshot_improvements
        return run_fewshot_improvements(config, data_dir, mapping, work)
    elif stage == "E8":
        from .improvements import run_improvements
        return run_improvements(config, data_dir, mapping, work)
    elif stage == "E7":
        _validate_final_config(config)
        source, group, seed = config["source"], config.get("group", "seven"), config["seed"]
        data = prepare_fewshot(data_dir, mapping, source, group, seed,
                               config.get("holdout_fraction", .4))
        model, medians, picked = fit_fewshot(data, config["k_per_class"],
                                             config["method"], config.get("alpha", .5),
                                             config["model"], config.get("jobs", 4))
        final_x, final_y = load_features(data_dir / DATA_FILES["iot_final"],
                                         load_mapping(mapping), group, "iot")
        scores = model.predict_proba(final_x.fillna(medians).to_numpy("float32"))[:, 1]
        m = metrics(final_y, scores, .5)
        rows.append({"stage": "E7", "source": source, "method": config["method"],
                     "model": config["model"], "seed": seed, "k_per_class": config["k_per_class"],
                     "target_labels_used": 0 if config["method"] == "source_only" else 2 * config["k_per_class"],
                     "alpha": config.get("alpha", .5) if config["method"] == "weighted" else None,
                     "n_final": len(final_y), "macro_f1": m["macro_f1"],
                     "balanced_accuracy": m["balanced_accuracy"], "auroc": m["auroc"],
                     "auprc": m["auprc"], "fpr": m["fpr"], "fnr": m["fnr"],
                     "total_cost": m["total_cost"], "cost_per_sample": m["cost_per_sample"],
                     "fp_cost": m["fp_cost"], "fn_cost": m["fn_cost"],
                     "attack_prevalence": m["attack_prevalence"],
                     "tn": m["confusion"]["tn"], "fp": m["confusion"]["fp"],
                     "fn": m["confusion"]["fn"], "tp": m["confusion"]["tp"]})
        (work / "selected_target_indices.json").write_text(json.dumps({
            "indices_from_iot_dev": picked.tolist(), "seed": seed}) + "\n", encoding="utf-8")
    else:
        sources = config.get("sources", ["2018"])
        group = config.get("group", "seven")
        holdout = config.get("holdout_fraction", .4)
        jobs = config.get("jobs", 4)
        model = config.get("model", "xgboost")
        for seed in config["seeds"]:
            for source in sources:
                data = prepare_fewshot(data_dir, mapping, source, group, seed, holdout)
                split = {"source": source, "seed": seed,
                         "source_train_indices": data.source_train_indices.tolist(),
                         "target_adapt_indices": data.adapt_indices.tolist(),
                         "target_holdout_indices": data.holdout_indices.tolist(),
                         "feature_overlap": overlap_report(data.x_adapt, data.x_holdout)}
                budgets = config["k_per_class"] if stage == "E4" else [config["k_per_class"]]
                # Check the largest budget before training any method for this split.
                data.target_sample(max(budgets))
                (work / f"split_{source}_s{seed}.json").write_text(
                    json.dumps(split) + "\n", encoding="utf-8")
                if stage == "E4":
                    rows.extend({"stage": stage, "source": source, **r} for r in
                                evaluate_fewshot(data, 0, "source_only", .5, model, jobs))
                    for k in config["k_per_class"]:
                        if k <= 0:
                            raise ValueError("E4 positive budgets only; zero source-only is automatic")
                        for method in config["methods"]:
                            if method == "source_only":
                                continue
                            rows.extend({"stage": stage, "source": source, **r} for r in
                                        evaluate_fewshot(data, k, method,
                                                         config.get("alpha", .5), model, jobs))
                elif stage == "E5":
                    for alpha in config["alphas"]:
                        rows.extend({"stage": stage, "source": source, **r} for r in
                                    evaluate_fewshot(data, config["k_per_class"], "weighted",
                                                     alpha, model, jobs))
                else:
                    for method in config["methods"]:
                        k = 0 if method == "source_only" else config["k_per_class"]
                        rows.extend({"stage": stage, "source": source, **r} for r in
                                    evaluate_fewshot(data, k, method, config.get("alpha", .5),
                                                     model, jobs, drifts=config["epsilons"]))
    if not rows:
        raise ValueError("Configuration produced no result rows")
    return pd.DataFrame(rows)


def _registry(results_dir: Path) -> dict:
    path = results_dir / "registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}, "active": {}, "events": []}


def _aggregate(stage: str, frame: pd.DataFrame) -> pd.DataFrame:
    if stage == "E0":
        return frame.copy()
    if stage in {"E1", "E2", "E3"}:
        keys = ["stage", "scenario", "source", "target", "model", "feature_group"]
    elif stage == "E7":
        return frame.copy()
    elif stage == "E8" or (stage == "E4" and "variant" in frame):
        keys = ["stage", "source", "variant", "method", "model", "k_per_class", "alpha",
                "feature_engineering", "tta", "evaluation_role", "threshold_strategy", "fp_cost", "fn_cost"]
        if "calibration" in frame:
            keys.append("calibration")
    else:
        keys = ["stage", "source", "method", "model", "k_per_class", "alpha", "epsilon", "direction"]
    values = [v for v in ("total_cost", "cost_per_sample", "always_benign_cost", "always_attack_cost",
                         "macro_f1", "balanced_accuracy", "auroc", "auprc", "fpr", "fnr",
                         "train_seconds", "inference_seconds", "tta_seconds", "ece", "brier_score", "log_loss",
                         "before_calibration_ece", "before_calibration_brier_score")
              if v in frame]
    return frame.groupby(keys, dropna=False)[values].agg(["mean", "std"]).reset_index().set_axis(
        keys + [f"{metric}_{stat}" for metric, stat in [(m, s) for m in values for s in ("mean", "std")]], axis=1)


def _activate(results_dir: Path, stage: str, run_id: str, registry: dict):
    entry = registry["runs"][run_id]
    if entry["stage"] != stage:
        raise ValueError(f"Run {run_id} belongs to {entry['stage']}, not {stage}")
    source = results_dir / "runs" / run_id / "summary.csv"
    if sha256(source) != entry["summary_sha256"]:
        raise ValueError("Snapshot CSV hash mismatch; refusing to activate")
    current = results_dir / "current"
    current.mkdir(exist_ok=True)
    temp = current / (stage + ".csv.tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, current / (stage + ".csv"))
    previous = registry["active"].get(stage)
    registry["active"][stage] = run_id
    registry["events"].append({"at_utc": datetime.now(timezone.utc).isoformat(),
                                "stage": stage, "from": previous, "to": run_id})
    _write_json(results_dir / "registry.json", registry)


def run_stage(stage: str, config_path: Path, data_dir: Path, results_dir: Path,
              mapping: Path) -> str:
    if stage not in STAGES:
        raise ValueError(stage)
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    _validate_config(stage, config, mapping)
    final_lock = _verify_final_lock(config_path, data_dir, mapping, config) if stage == "E7" else None
    if data_dir.resolve() == results_dir.resolve():
        raise ValueError("Data and results directories must differ")
    # Validate and fingerprint only actual inputs before expensive training.
    inputs = _input_hashes(data_dir, stage, mapping, config)
    code_hashes = _code_hashes()
    if final_lock is not None and (code_hashes != final_lock["code_sha256"] or
            {k: v for k, v in inputs.items() if k != "iot_final"} != final_lock["inputs_sha256"]):
        raise ValueError("E7 inputs or code changed after lock verification")
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{stage}_{uuid.uuid4().hex[:8]}"
    if stage == "E7":
        # Stored beside the dataset so changing results-dir does not reset it.
        # Keep attempted runs too: a failure may happen after predictions exist.
        claim_final_evaluation(data_dir / DATA_FILES["iot_final"],
                               {"run_id": run_id, "results_dir": str(results_dir.resolve()),
                                "selection": final_lock})
    work = Path(tempfile.mkdtemp(prefix=f".running_{stage}_", dir=runs_dir))
    snapshot = runs_dir / run_id
    (work / "config.json").write_text(config_text, encoding="utf-8")
    if final_lock is not None:
        _write_json(work / "selection_lock.json", final_lock)
    frame = _produce(stage, config, data_dir, mapping, work)
    if inputs != _input_hashes(data_dir, stage, mapping, config) or code_hashes != _code_hashes():
        raise ValueError("Input data, mapping or code changed during execution; results were not activated")
    frame.to_csv(work / "summary.csv", index=False)
    _aggregate(stage, frame).to_csv(work / "aggregate.csv", index=False)
    manifest = {"run_id": run_id, "stage": stage,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "configuration_sha256": sha256(work / "config.json"),
                "inputs_sha256": inputs,
                "code_sha256": code_hashes,
                "summary_sha256": sha256(work / "summary.csv"),
                "aggregate_sha256": sha256(work / "aggregate.csv"),
                "rows": len(frame), "versions": {p: importlib.metadata.version(p)
                                                   for p in ("numpy", "pandas", "scikit-learn", "xgboost")},
                "selection_metric": config.get("selection_metric", "total_cost" if stage == "E8" else "macro_f1"),
                "artifact_sha256": {str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob("*")) if p.is_file()},
                "protocol_note": "E0-E6 and E8 never read IoT FINAL_TEST. E8 marks TTA as transductive. E7 requires a locked preselected config; prior research has already examined related data."}
    if stage == "E8" and any(v["method"] == "hybrid" for v in config["variants"]):
        manifest["versions"]["torch"] = importlib.metadata.version("torch")
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(work, snapshot)
    with _exclusive_registry(results_dir):
        registry = _registry(results_dir)
        registry["runs"][run_id] = {"stage": stage, "summary_sha256": manifest["summary_sha256"],
                                     "created_utc": manifest["created_utc"], "rows": len(frame)}
        _activate(results_dir, stage, run_id, registry)
    return run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--results-dir", type=Path, default=ROOT / "results")
    run.add_argument("--mapping", type=Path, default=ROOT / "config" / "feature_mapping.json")
    listing = sub.add_parser("list")
    listing.add_argument("--results-dir", type=Path, default=ROOT / "results")
    locking = sub.add_parser("lock-final", help="Record E7 selection without reading FINAL")
    locking.add_argument("--config", type=Path, required=True)
    locking.add_argument("--data-dir", type=Path, required=True)
    locking.add_argument("--mapping", type=Path, default=ROOT / "config" / "feature_mapping.json")
    locking.add_argument("--reason", required=True)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--stage", choices=STAGES, required=True)
    rollback.add_argument("--run-id", required=True)
    rollback.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if args.action == "run":
        config = args.config or ROOT / "config" / STAGE_CONFIGS[args.stage]
        run_id = run_stage(args.stage, config, args.data_dir, args.results_dir, args.mapping)
        print(json.dumps({"run_id": run_id, "csv": str(args.results_dir / "current" / f"{args.stage}.csv")}, indent=2))
    elif args.action == "list":
        print(json.dumps(_registry(args.results_dir), indent=2))
    elif args.action == "lock-final":
        print(json.dumps({"lock": str(lock_final(args.config, args.data_dir, args.mapping, args.reason))}))
    else:
        with _exclusive_registry(args.results_dir):
            registry = _registry(args.results_dir)
            if args.run_id not in registry["runs"]:
                raise ValueError("Unknown run ID")
            _activate(args.results_dir, args.stage, args.run_id, registry)
        print(json.dumps({"active": args.run_id, "csv": str(args.results_dir / "current" / f"{args.stage}.csv")}, indent=2))


if __name__ == "__main__":
    main()
