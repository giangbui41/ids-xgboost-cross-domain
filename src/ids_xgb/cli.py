"""Command line interface. No command automatically evaluates FINAL_TEST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_features, load_mapping
from .experiment import MODEL_TYPES, evaluate, freeze, save_json, train
from .prepare import prepare_iot, prepare_source
from .matrix import SCENARIOS, run_matrix, run_sensitivity
from .robustness import run_drift

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = PROJECT / "config" / "feature_mapping.json"


def audit(mapping_path: Path, source17: Path, source18: Path, iot_dev: Path,
          output: Path) -> dict:
    mapping = load_mapping(mapping_path)
    report = {"mapping_version": mapping["mapping_version"],
              "warning": "Numerical consistency cannot establish semantic equivalence. DEV labels are diagnostic only.",
              "datasets": {}}
    for domain, path in [("2017", source17), ("2018", source18), ("iot", iot_dev)]:
        x, y = load_features(path, mapping, "seven", domain)
        feature_stats = {}
        for feature in x.columns:
            values = x[feature]
            feature_stats[feature] = {
                "column": mapping["features"][feature][domain],
                "missing_fraction": float(values.isna().mean()),
                "zero_fraction": float(values.eq(0).mean()),
                "median": float(values.median()), "p95": float(values.quantile(.95)),
                "benign_median": float(values[y == 0].median()),
                "attack_median": float(values[y == 1].median()),
            }
        a = x["flow_packets_per_second"]
        b = x["forward_packets_per_second"] + x["backward_packets_per_second"]
        consistent = np.isclose(a, b, atol=1e-3, rtol=1e-3)
        ordered = (x["packet_length_min"] <= x["packet_length_mean"]) & (
            x["packet_length_mean"] <= x["packet_length_max"])
        report["datasets"][domain] = {
            "path_name": path.name, "rows": len(y), "attack_prevalence": float(y.mean()),
            "rate_identity_fraction": float(consistent.mean()),
            "packet_length_order_fraction": float(ordered.mean()),
            "features": feature_stats,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    save_json(output, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-source", help="Sample official 2017/2018 CSVs without balancing")
    p.add_argument("--domain", choices=["2017", "2018"], required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)

    p2 = sub.add_parser("prepare-iot", help="Sample IoT CSVs and make a new DEV/FINAL split")
    p2.add_argument("--raw-dir", type=Path, required=True)
    p2.add_argument("--dev-output", type=Path, required=True)
    p2.add_argument("--final-output", type=Path, required=True)
    p2.add_argument("--sample-frac", type=float, default=0.05)
    p2.add_argument("--seed", type=int, default=20260819)
    p2.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p2.add_argument("--deduplicate", action="store_true",
                    help="Remove identical full feature vectors before splitting; changes the data distribution")

    a = sub.add_parser("audit", help="Inspect mapped source data and IoT DEV only")
    a.add_argument("--source-2017", type=Path, required=True)
    a.add_argument("--source-2018", type=Path, required=True)
    a.add_argument("--iot-dev", type=Path, required=True)
    a.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    a.add_argument("--output", type=Path, required=True)

    t = sub.add_parser("train", help="Train source-only XGBoost; no IoT data is read")
    t.add_argument("--source", choices=["2017", "2018"], required=True)
    t.add_argument("--source-csv", type=Path, required=True)
    t.add_argument("--group", choices=["seven", "without_backward", "length_only"], required=True)
    t.add_argument("--model", choices=MODEL_TYPES, default="xgboost")
    t.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--jobs", type=int, default=4)

    e = sub.add_parser("evaluate-dev", help="Diagnostic evaluation on IoT DEV")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--iot-dev", type=Path, required=True)
    e.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    e.add_argument("--output", type=Path, required=True)

    f = sub.add_parser("freeze", help="Lock one selected model before FINAL evaluation")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)

    z = sub.add_parser("evaluate-final", help="One locked evaluation on untouched IoT FINAL_TEST")
    z.add_argument("--frozen", type=Path, required=True)
    z.add_argument("--iot-final", type=Path, required=True)
    z.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    z.add_argument("--output", type=Path, required=True)

    comp = sub.add_parser("compare", help="Build a comparable table from DEV evaluations")
    comp.add_argument("--runs", type=Path, nargs="+", required=True)
    comp.add_argument("--output", type=Path, required=True)

    matrix = sub.add_parser("matrix", help="Run E1--E4 in-domain and cross-domain benchmark")
    matrix.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    matrix.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=list(MODEL_TYPES))
    matrix.add_argument("--data-dir", type=Path, required=True)
    matrix.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    matrix.add_argument("--group", choices=["seven", "without_backward", "length_only"], default="seven")
    matrix.add_argument("--seed", type=int, default=42)
    matrix.add_argument("--jobs", type=int, default=4)
    matrix.add_argument("--out", type=Path, required=True)

    sens = sub.add_parser("sensitivity", help="E5 analogue: XGBoost tree-depth sweep on E3 2018-to-IoT")
    sens.add_argument("--data-dir", type=Path, required=True)
    sens.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    sens.add_argument("--group", choices=["seven", "without_backward", "length_only"], default="seven")
    sens.add_argument("--depths", nargs="+", type=int, default=[3, 6, 9])
    sens.add_argument("--seed", type=int, default=42)
    sens.add_argument("--jobs", type=int, default=4)
    sens.add_argument("--out", type=Path, required=True)

    drift = sub.add_parser("drift", help="E6 analogue: label-free measurement drift on IoT DEV")
    drift.add_argument("--run", type=Path, required=True)
    drift.add_argument("--iot-dev", type=Path, required=True)
    drift.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    drift.add_argument("--epsilons", nargs="+", type=float, default=[0, .05, .1, .2])
    drift.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare-source":
        result = prepare_source(args.raw_dir, args.output, args.mapping,
                                args.domain, args.sample_frac, args.seed)
    elif args.command == "prepare-iot":
        result = prepare_iot(args.raw_dir, args.dev_output, args.final_output,
                             args.mapping, args.sample_frac, args.seed, deduplicate=args.deduplicate)
    elif args.command == "audit":
        result = audit(args.mapping, args.source_2017, args.source_2018,
                       args.iot_dev, args.output)
    elif args.command == "train":
        result = train(args.source, args.source_csv, args.mapping, args.group,
                       args.out, args.seed, args.jobs, args.model)
    elif args.command == "evaluate-dev":
        result = evaluate(args.run, args.iot_dev, args.mapping, "dev", args.output)
    elif args.command == "freeze":
        result = freeze(args.run, args.out)
    elif args.command == "compare":
        if args.output.exists():
            raise FileExistsError(args.output)
        evaluations = []
        for path in args.runs:
            item = json.loads((path / "iot_dev.json").read_text(encoding="utf-8"))
            run_meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
            for key in ("protocol", "mapping_sha256", "seed", "model_type"):
                item.setdefault(key, run_meta.get(key, "xgboost" if key == "model_type" else None))
            evaluations.append(item)
        if any(e["stage"] != "dev" for e in evaluations):
            raise ValueError("compare accepts DEV results only")
        for key in ("target_sha256", "protocol", "mapping_sha256", "seed"):
            if len({e[key] for e in evaluations}) != 1:
                raise ValueError(f"Comparison requires the same {key}")
        rows = []
        for path, item in zip(args.runs, evaluations):
            m = item["metrics"]
            rows.append({"run": str(path), "source": item["source"],
                         "model": item["model_type"], "features": item["group"],
                         "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                         "auroc": m["auroc"], "auprc": m["auprc"],
                         "fpr": m["fpr"], "fnr": m["fnr"],
                         "attack_prevalence": m["attack_prevalence"],
                         "accepted_fraction": m["accepted_fraction"]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        result = {"output": str(args.output), "rows": rows,
                  "note": "Diagnostic comparison. No DEV result should be mistaken for a new independent final test."}
    elif args.command == "matrix":
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        result = {"summary": str(args.out / "summary.csv"),
                  "rows": run_matrix(names, args.models, args.data_dir,
                                     args.mapping, args.group, args.seed, args.jobs, args.out)}
    elif args.command == "sensitivity":
        result = {"rows": run_sensitivity(args.data_dir, args.mapping, args.group,
                                           args.seed, args.jobs, args.depths, args.out)}
    elif args.command == "drift":
        result = {"rows": run_drift(args.run, args.iot_dev, args.mapping,
                                    args.epsilons, args.output)}
    else:
        result = evaluate(args.frozen, args.iot_final, args.mapping, "final", args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
