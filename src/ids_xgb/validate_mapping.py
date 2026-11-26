"""Semantic mapping validation for the 7 canonical features.

Compares distributions (not just names) between CIC-IDS2017/2018 and CICIoT2023.
Outputs JSON + Markdown report for paper appendix.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .data import load_mapping, load_features, sha256


EXCLUDED_PATTERN = re.compile(
    r"(^|_)(flow_?id|timestamp|src_?ip|dst_?ip|source_?ip|destination_?ip)($|_)"
)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_native_frame(path: Path) -> pd.DataFrame:
    """Load CSV, drop Label and excluded columns, coerce to numeric."""
    frame = pd.read_csv(path, low_memory=False)
    if "Label" not in frame:
        raise ValueError(f"Missing Label in {path}")
    frame = frame.drop(columns=["Label"])
    keep = [c for c in frame if not EXCLUDED_PATTERN.search(normalize_name(c))]
    x = frame[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x


def compute_stats(series: pd.Series) -> dict[str, Any]:
    """Compute distribution statistics for a single feature."""
    vals = series.dropna().to_numpy()
    if len(vals) == 0:
        return {"count": 0, "missing_fraction": 1.0}
    return {
        "count": int(len(vals)),
        "missing_fraction": float(series.isna().mean()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
        "zeros_fraction": float((vals == 0).mean()),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
    }


def ks_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Two-sample KS test with effect size."""
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    if len(a_clean) < 2 or len(b_clean) < 2:
        return {"statistic": np.nan, "pvalue": np.nan, "effect_size": np.nan}
    stat, p = stats.ks_2samp(a_clean, b_clean, alternative="two-sided", mode="auto")
    # Effect size: max absolute difference in ECDF
    return {"statistic": float(stat), "pvalue": float(p), "effect_size": float(stat)}


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    if len(a_clean) == 0 or len(b_clean) == 0:
        return np.nan
    return float(stats.wasserstein_distance(a_clean, b_clean))


def mean_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Ratio of means (a/b), handling zeros."""
    a_mean = a[np.isfinite(a)].mean()
    b_mean = b[np.isfinite(b)].mean()
    if b_mean == 0:
        return np.inf if a_mean != 0 else 1.0
    return float(a_mean / b_mean)


def validate_mapping(
    mapping_path: Path,
    data_dir: Path,
    group: str = "canonical_7",
    sample_size: int | None = 50000,
    ks_threshold: float = 0.3,
    mean_ratio_threshold: float = 10.0,
    zero_fraction_threshold: float = 0.95,
) -> dict[str, Any]:
    """
    Validate semantic equivalence of mapped features across domains.

    Args:
        mapping_path: Path to feature_mapping.json
        data_dir: Directory containing the 4 CSV files
        group: Feature group to validate (default: canonical_7)
        sample_size: Subsample size per domain for speed (None = full)
        ks_threshold: KS statistic > this = potential semantic mismatch
        mean_ratio_threshold: Mean ratio > this = potential unit/scale mismatch
        zero_fraction_threshold: Zero fraction > this = near-constant feature

    Returns:
        Dict with per-feature validation results and overall summary.
    """
    mapping = load_mapping(mapping_path)
    if group not in mapping["groups"]:
        raise ValueError(f"Group {group} not in mapping. Available: {list(mapping['groups'].keys())}")

    canonical_features = mapping["groups"][group]

    # Data file paths
    data_files = {
        "2017": data_dir / "cicids2017_balanced_77feats.csv",
        "2018": data_dir / "cleaned_data_sampled.csv",
        "iot": data_dir / "ciciot2023_dev.csv",
    }

    # Load native frames (all numeric columns)
    native_frames = {}
    for domain, path in data_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        native_frames[domain] = load_native_frame(path)

    # Subsample if requested
    if sample_size is not None:
        for domain in native_frames:
            if len(native_frames[domain]) > sample_size:
                native_frames[domain] = native_frames[domain].sample(
                    n=sample_size, random_state=42
                )

    # Build alias maps from mapping
    aliases = {}
    for domain in ("2017", "2018", "iot"):
        aliases[domain] = {}
        for canonical, spec in mapping["features"].items():
            if canonical in canonical_features:
                src_col = spec[domain]
                aliases[domain][normalize_name(src_col)] = canonical

    # Resolve column names in each native frame
    resolved_columns = {}
    for domain in ("2017", "2018", "iot"):
        frame = native_frames[domain]
        col_map = {}
        for col in frame.columns:
            norm = normalize_name(col)
            if norm in aliases[domain]:
                canonical = aliases[domain][norm]
                if canonical in col_map:
                    raise ValueError(f"Ambiguous mapping for {canonical} in {domain}: {col_map[canonical]} vs {col}")
                col_map[canonical] = col
        resolved_columns[domain] = col_map

    # Validate each canonical feature
    results = {}
    warnings = []

    for canonical in canonical_features:
        # Check all three domains have this feature
        missing_domains = [d for d in ("2017", "2018", "iot") if canonical not in resolved_columns[d]]
        if missing_domains:
            results[canonical] = {
                "status": "missing_in_domains",
                "missing_domains": missing_domains,
            }
            warnings.append(f"{canonical}: missing in {missing_domains}")
            continue

        # Extract series
        series = {}
        for domain in ("2017", "2018", "iot"):
            col = resolved_columns[domain][canonical]
            series[domain] = native_frames[domain][col]

        # Compute stats
        stats_dict = {d: compute_stats(series[d]) for d in ("2017", "2018", "iot")}

        # Pairwise comparisons
        comparisons = {}
        for src, tgt in [("2017", "iot"), ("2018", "iot"), ("2017", "2018")]:
            a = series[src].dropna().to_numpy()
            b = series[tgt].dropna().to_numpy()
            ks = ks_test(a, b)
            wdist = wasserstein_distance(a, b)
            mratio = mean_ratio(a, b)
            comparisons[f"{src}_vs_{tgt}"] = {
                "ks_statistic": ks["statistic"],
                "ks_pvalue": ks["pvalue"],
                "ks_effect_size": ks["effect_size"],
                "wasserstein_distance": wdist,
                "mean_ratio": mratio,
            }

        # Determine status
        status = "ok"
        feature_warnings = []

        # Check zero fraction (near-constant)
        for domain in ("2017", "2018", "iot"):
            zf = stats_dict[domain].get("zeros_fraction", 0)
            if zf > zero_fraction_threshold:
                status = "warning"
                feature_warnings.append(f"{domain}: zero_fraction={zf:.3f} > {zero_fraction_threshold}")

        # Check KS drift (2017 vs IoT, 2018 vs IoT)
        for pair in ("2017_vs_iot", "2018_vs_iot"):
            ks_stat = comparisons[pair]["ks_statistic"]
            if not np.isnan(ks_stat) and ks_stat > ks_threshold:
                status = "warning"
                feature_warnings.append(f"{pair}: KS={ks_stat:.3f} > {ks_threshold}")

        # Check mean ratio (unit/scale mismatch)
        for pair in ("2017_vs_iot", "2018_vs_iot"):
            mratio = comparisons[pair]["mean_ratio"]
            if not np.isnan(mratio) and (mratio > mean_ratio_threshold or mratio < 1/mean_ratio_threshold):
                status = "warning"
                feature_warnings.append(f"{pair}: mean_ratio={mratio:.2f} outside [1/{mean_ratio_threshold}, {mean_ratio_threshold}]")

        results[canonical] = {
            "status": status,
            "source_columns": {d: resolved_columns[d][canonical] for d in ("2017", "2018", "iot")},
            "statistics": stats_dict,
            "comparisons": comparisons,
            "warnings": feature_warnings,
        }
        warnings.extend([f"{canonical}: {w}" for w in feature_warnings])

    # Overall summary
    status_counts = {"ok": 0, "warning": 0, "missing_in_domains": 0}
    for r in results.values():
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    summary = {
        "mapping_version": mapping.get("mapping_version", "unknown"),
        "mapping_sha256": sha256(mapping_path),
        "group": group,
        "canonical_features": canonical_features,
        "sample_size": sample_size,
        "thresholds": {
            "ks_threshold": ks_threshold,
            "mean_ratio_threshold": mean_ratio_threshold,
            "zero_fraction_threshold": zero_fraction_threshold,
        },
        "status_counts": status_counts,
        "overall_status": "pass" if status_counts["warning"] == 0 and status_counts["missing_in_domains"] == 0 else "review_required",
        "warnings": warnings,
    }

    return {
        "summary": summary,
        "features": results,
    }


def generate_markdown_report(validation_result: dict[str, Any]) -> str:
    """Generate human-readable Markdown report."""
    summary = validation_result["summary"]
    features = validation_result["features"]

    lines = [
        "# Semantic Mapping Validation Report",
        "",
        f"**Mapping version:** {summary['mapping_version']}  ",
        f"**Mapping SHA256:** {summary['mapping_sha256']}  ",
        f"**Feature group:** {summary['group']}  ",
        f"**Sample size per domain:** {summary['sample_size'] or 'full'}  ",
        f"**Overall status:** {summary['overall_status'].upper()}  ",
        "",
        "## Thresholds Used",
        "",
        f"- KS statistic threshold: {summary['thresholds']['ks_threshold']}",
        f"- Mean ratio threshold: {summary['thresholds']['mean_ratio_threshold']} (and reciprocal)",
        f"- Zero fraction threshold: {summary['thresholds']['zero_fraction_threshold']}",
        "",
        "## Summary Counts",
        "",
        f"- OK: {summary['status_counts'].get('ok', 0)}",
        f"- Warning: {summary['status_counts'].get('warning', 0)}",
        f"- Missing in domains: {summary['status_counts'].get('missing_in_domains', 0)}",
        "",
    ]

    if summary["warnings"]:
        lines.append("## ⚠️ Warnings")
        lines.append("")
        for w in summary["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Per-Feature Details")
    lines.append("")

    for canonical, result in features.items():
        lines.append(f"### {canonical}")
        lines.append("")
        lines.append(f"**Status:** {result['status'].upper()}  ")
        lines.append(f"**Source columns:**  ")
        for d, col in result["source_columns"].items():
            lines.append(f"  - {d}: `{col}`")
        lines.append("")

        if result["status"] == "missing_in_domains":
            lines.append(f"**Missing in:** {result['missing_domains']}")
            lines.append("")
            continue

        # Statistics table
        lines.append("**Statistics:**")
        lines.append("")
        lines.append("| Domain | Count | Missing% | Mean | Std | Min | Max | Median | Zeros% |")
        lines.append("|--------|-------|----------|------|-----|-----|-----|--------|--------|")
        for domain in ("2017", "2018", "iot"):
            s = result["statistics"][domain]
            lines.append(
                f"| {domain} | {s['count']} | {s['missing_fraction']:.2%} | "
                f"{s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
                f"{s['median']:.4f} | {s['zeros_fraction']:.2%} |"
            )
        lines.append("")

        # Comparisons table
        lines.append("**Pairwise Comparisons:**")
        lines.append("")
        lines.append("| Comparison | KS Stat | KS p-value | KS Effect | Wasserstein | Mean Ratio |")
        lines.append("|------------|---------|------------|-----------|-------------|------------|")
        for comp_name, comp in result["comparisons"].items():
            lines.append(
                f"| {comp_name} | {comp['ks_statistic']:.4f} | {comp['ks_pvalue']:.4f} | "
                f"{comp['ks_effect_size']:.4f} | {comp['wasserstein_distance']:.4f} | {comp['mean_ratio']:.4f} |"
            )
        lines.append("")

        if result["warnings"]:
            lines.append("**⚠️ Feature Warnings:**")
            for w in result["warnings"]:
                lines.append(f"- {w}")
            lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate semantic mapping of 7 canonical features")
    parser.add_argument("--mapping", type=Path, required=True, help="Path to feature_mapping.json")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing CSV files")
    parser.add_argument("--group", default="canonical_7", help="Feature group to validate")
    parser.add_argument("--sample-size", type=int, default=50000, help="Subsample size (0 = full)")
    parser.add_argument("--ks-threshold", type=float, default=0.3, help="KS statistic warning threshold")
    parser.add_argument("--mean-ratio-threshold", type=float, default=10.0, help="Mean ratio warning threshold")
    parser.add_argument("--zero-threshold", type=float, default=0.95, help="Zero fraction warning threshold")
    parser.add_argument("--output-json", type=Path, help="Output JSON report path")
    parser.add_argument("--output-md", type=Path, help="Output Markdown report path")
    args = parser.parse_args()

    sample_size = args.sample_size if args.sample_size > 0 else None

    result = validate_mapping(
        mapping_path=args.mapping,
        data_dir=args.data_dir,
        group=args.group,
        sample_size=sample_size,
        ks_threshold=args.ks_threshold,
        mean_ratio_threshold=args.mean_ratio_threshold,
        zero_fraction_threshold=args.zero_threshold,
    )

    # Print summary to stdout
    summary = result["summary"]
    print(f"Overall status: {summary['overall_status']}")
    print(f"OK: {summary['status_counts'].get('ok', 0)}, "
          f"Warnings: {summary['status_counts'].get('warning', 0)}, "
          f"Missing: {summary['status_counts'].get('missing_in_domains', 0)}")
    if summary["warnings"]:
        print("\nWarnings:")
        for w in summary["warnings"]:
            print(f"  - {w}")

    # Write JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nJSON report written to: {args.output_json}")

    # Write Markdown
    if args.output_md:
        md = generate_markdown_report(result)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")
        print(f"Markdown report written to: {args.output_md}")


if __name__ == "__main__":
    main()