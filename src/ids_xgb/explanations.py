"""Bounded native XGBoost TreeSHAP exports without the external shap package."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix


NOTE = (
    "Source-target attribution differences describe model behavior on the sampled domains; "
    "they do not establish causal bias or validate feature-mapping semantics. Contributions "
    "sum to the raw log-odds margin, not to probability. Scores shown belong to the tree "
    "head before any subsequent temperature transformation. For a DANN hybrid, latent_* "
    "or dann_embedding_* features explain only the XGBoost head, not the original input features."
)


def _sample_explanations(booster, matrix, feature_names, rng, max_samples, iteration_range):
    positions = np.sort(rng.choice(len(matrix), size=min(len(matrix), max_samples), replace=False))
    sampled = matrix[positions]
    dmatrix = DMatrix(sampled, feature_names=booster.feature_names)
    contributions = np.asarray(booster.predict(
        dmatrix, pred_contribs=True, approx_contribs=False, iteration_range=iteration_range))
    margin = np.asarray(booster.predict(dmatrix, output_margin=True, iteration_range=iteration_range))
    scores = np.asarray(booster.predict(dmatrix, iteration_range=iteration_range))
    if contributions.shape != (len(positions), len(feature_names) + 1) or margin.ndim != 1:
        raise ValueError("TreeSHAP exporter supports binary classification only")
    reconstructed = contributions.astype(np.float64).sum(axis=1)
    if not (np.isfinite(contributions).all() and np.isfinite(margin).all()
            and np.isfinite(scores).all()):
        raise ValueError("Nonfinite TreeSHAP contributions or predictions")
    if not np.allclose(reconstructed, margin, atol=1e-4, rtol=1e-4):
        raise ValueError("TreeSHAP additivity check failed against model raw margins")
    result = pd.DataFrame({"row_position": positions, "probability_attack": scores,
                           "raw_margin": margin, "bias": contributions[:, -1],
                           "additivity_error": reconstructed - margin})
    for column, name in enumerate(feature_names):
        result["value__" + name] = sampled[:, column]
        result["shap__" + name] = contributions[:, column]
    return result, contributions[:, :-1]


def _summary_svg(comparison: pd.DataFrame) -> str:
    rows = comparison.sort_values("target_mean_abs_shap", ascending=False)
    width, height, label_width = 1000, 70 + 42 * len(rows), 370
    scale = max(float(rows[["source_mean_abs_shap", "target_mean_abs_shap"]].max().max()), 1e-12)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Source and target mean absolute SHAP contributions">',
             '<text x="370" y="20">Mean absolute contribution (log-odds)</text>',
             '<rect x="370" y="32" width="14" height="10" fill="#2563eb"/><text x="390" y="42">Source</text>',
             '<rect x="480" y="32" width="14" height="10" fill="#d97706"/><text x="500" y="42">Target</text>']
    for row, (_, values) in enumerate(rows.iterrows()):
        y = 65 + 42 * row
        parts.append(f'<text x="6" y="{y + 11}" font-size="12">{html.escape(str(values["feature"]))}</text>')
        for offset, domain, color in [(0, "source", "#2563eb"), (15, "target", "#d97706")]:
            value = float(values[f"{domain}_mean_abs_shap"])
            bar_width = value / scale * 500
            parts.append(f'<rect x="{label_width}" y="{y + offset}" width="{bar_width:.2f}" height="11" fill="{color}"/>')
            parts.append(f'<text x="{label_width + bar_width + 5:.2f}" y="{y + offset + 10}" font-size="11">{value:.4g}</text>')
    return "".join(parts) + "</svg>"


def _write_html(path: Path, comparison: pd.DataFrame, samples: dict, feature_names: list[str]):
    payload = {"features": feature_names, "domains": {}}
    for domain, frame in samples.items():
        # JSON null represents an original missing value without executing any
        # feature names or content as HTML/JavaScript.
        payload["domains"][domain] = json.loads(frame.to_json(orient="records"))
    serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    table = comparison.to_html(index=False, float_format=lambda value: f"{value:.5g}", escape=True)
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TreeSHAP source-target diagnostics</title>
<style>body{font:15px system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 18px;color:#172033}
svg{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}
.scroll{overflow-x:auto}select{padding:7px;margin:8px}p{line-height:1.5}.positive{color:#c2410c}.negative{color:#1d4ed8}
details{margin:18px 0}#bars{margin:16px 0}.barrow{display:grid;grid-template-columns:minmax(200px,1fr) 2fr 90px;align-items:center;gap:8px;margin:5px 0}
.track{height:12px;background:#eef2f6}.bar{height:100%}</style>
<h1>TreeSHAP source-target diagnostics</h1>
<p>__NOTE__</p><p>Both domains are sampled independently without labels. All samples are identified by their position in the arrays supplied to this exporter.</p>
__SVG__<details><summary>Domain contribution statistics</summary><div class="scroll">__TABLE__</div></details>
<h2>Individual decision</h2><label>Domain <select id="domain"><option value="source">Source</option><option value="target">Target</option></select></label>
<label>Sample <select id="sample"></select></label><p id="prediction"></p>
<p>Orange contributions increase the attack log-odds; blue contributions decrease them. Bars show magnitude; the sign is printed separately.</p>
<div id="bars"></div><details><summary>Input values and every contribution</summary><div class="scroll"><table><thead><tr><th>Feature</th><th>Value</th><th>Contribution</th></tr></thead><tbody id="values"></tbody></table></div></details>
<script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const domain=document.getElementById('domain'),sample=document.getElementById('sample');
function draw(){const row=data.domains[domain.value][Number(sample.value)];
document.getElementById('prediction').textContent='P(attack) = '+row.probability_attack.toFixed(5)+'; raw margin = '+row.raw_margin.toFixed(5)+'; baseline = '+row.bias.toFixed(5)+'; additivity error = '+row.additivity_error.toExponential(2);
const ordered=data.features.map(f=>({name:f,value:row['value__'+f],shap:row['shap__'+f]})).sort((a,b)=>Math.abs(b.shap)-Math.abs(a.shap));
const bars=document.getElementById('bars'),tbody=document.getElementById('values');bars.replaceChildren();tbody.replaceChildren();
const max=Math.max(...ordered.map(f=>Math.abs(f.shap)),1e-12);
for(const f of ordered){const tr=document.createElement('tr');for(const v of [f.name,f.value===null?'missing':f.value.toPrecision(6),f.shap.toPrecision(6)]){const td=document.createElement('td');td.textContent=v;tr.append(td);}tbody.append(tr);}
for(const f of ordered.slice(0,15)){const line=document.createElement('div');line.className='barrow';const name=document.createElement('span');name.textContent=f.name;const track=document.createElement('div');track.className='track';const bar=document.createElement('div');bar.className='bar';bar.style.width=(100*Math.abs(f.shap)/max)+'%';bar.style.background=f.shap>=0?'#c2410c':'#1d4ed8';track.append(bar);const value=document.createElement('span');value.textContent=(f.shap>=0?'+':'')+f.shap.toFixed(4);line.append(name,track,value);bars.append(line);}}
function chooseDomain(){sample.replaceChildren();data.domains[domain.value].forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent='row '+row.row_position;sample.append(option);});draw();}
domain.addEventListener('change',chooseDomain);sample.addEventListener('change',draw);chooseDomain();
</script></html>"""
    for token, value in [("__NOTE__", html.escape(NOTE)), ("__SVG__", _summary_svg(comparison)),
                         ("__TABLE__", table), ("__DATA__", serialized)]:
        document = document.replace(token, value)
    path.write_text(document, encoding="utf-8")


def export_tree_explanations(model, source_x: np.ndarray, target_x: np.ndarray,
                             feature_names: list[str], out_dir: Path,
                             seed: int = 42, max_samples: int = 256) -> dict:
    """Export exact native TreeSHAP contributions and an offline HTML report.

    Accepts a binary-logistic XGBClassifier or Booster. For a hybrid, pass the
    fitted XGBoost head and the source/target latent matrices, naming columns
    ``latent_0``, ``latent_1`` or ``dann_embedding_00``, etc. No target labels are
    accepted or consumed. Returned file paths are relative to ``out_dir`` so
    they remain valid when a staging directory is moved into a run snapshot.
    With early stopping, the same best iteration used by XGBClassifier is used.
    ``row_position`` identifies input-array positions, not original CSV row IDs.
    """
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if not feature_names or len(set(feature_names)) != len(feature_names) or any(
            not isinstance(name, str) for name in feature_names):
        raise ValueError("Provide unique nonempty feature names")
    booster = model if isinstance(model, Booster) else getattr(model, "get_booster", lambda: None)()
    if not isinstance(booster, Booster):
        raise ValueError("Native TreeSHAP requires an XGBoost classifier or Booster")
    config = json.loads(booster.save_config())["learner"]
    if config["learner_train_param"]["objective"] != "binary:logistic":
        raise ValueError("TreeSHAP exporter requires binary:logistic attack probabilities")
    if config["gradient_booster"]["name"] not in {"gbtree", "dart"}:
        raise ValueError("TreeSHAP requires a tree-based booster")
    if booster.num_features() != len(feature_names):
        raise ValueError("Explanation feature count does not match the fitted model")
    if booster.feature_names is not None and list(booster.feature_names) != list(feature_names):
        raise ValueError("Explanation feature names/order do not match the fitted model")
    arrays = {}
    for domain, matrix in [("source", source_x), ("target", target_x)]:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names) or not len(matrix):
            raise ValueError(f"{domain} explanation input must be a nonempty matching feature matrix")
        if np.isinf(matrix).any():
            raise ValueError("Explanation arrays may contain missing values but not infinity")
        arrays[domain] = matrix
    best_iteration = booster.attr("best_iteration")
    iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
    generators = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    samples, contributions = {}, {}
    for (domain, matrix), rng in zip(arrays.items(), generators):
        samples[domain], contributions[domain] = _sample_explanations(
            booster, matrix, feature_names, rng, max_samples, iteration_range)
    comparison = pd.DataFrame({"feature": feature_names})
    for domain, values in contributions.items():
        comparison[f"{domain}_mean_abs_shap"] = np.abs(values).mean(axis=0)
        comparison[f"{domain}_mean_signed_shap"] = values.mean(axis=0)
    comparison["delta_mean_abs_target_minus_source"] = (
        comparison["target_mean_abs_shap"] - comparison["source_mean_abs_shap"])
    comparison["delta_mean_signed_target_minus_source"] = (
        comparison["target_mean_signed_shap"] - comparison["source_mean_signed_shap"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {domain + "_contributions": out_dir / (domain + "_contributions.csv") for domain in samples}
    for domain, frame in samples.items():
        frame.to_csv(paths[domain + "_contributions"], index=False)
    paths["domain_comparison"] = out_dir / "domain_comparison.csv"
    comparison.to_csv(paths["domain_comparison"], index=False)
    paths["report"] = out_dir / "explanations.html"
    _write_html(paths["report"], comparison, samples, list(feature_names))
    summary = {
        "method": "native_tree_shap", "output_scale": "raw_log_odds", "seed": seed,
        "max_samples_per_domain": max_samples, "feature_names": list(feature_names),
        "representation": "latent" if all(name.startswith(("latent_", "dann_embedding_")) for name in feature_names) else "input",
        "iteration_range": list(iteration_range), "note": NOTE,
        "source_samples": len(samples["source"]), "target_samples": len(samples["target"]),
        "max_additivity_error": float(max(frame.additivity_error.abs().max() for frame in samples.values())),
        "source_row_positions": samples["source"].row_position.tolist(),
        "target_row_positions": samples["target"].row_position.tolist(),
        "files": {name: path.name for name, path in paths.items()},
    }
    (out_dir / "explanations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
