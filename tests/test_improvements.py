"""Integration and protocol checks for cost-sensitive E8 variants."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_xgb.data import load_mapping
from ids_xgb.experiment import metrics, select_cost_threshold
from ids_xgb.fewshot import FewshotData
from ids_xgb.improvements import run_improvements, validate_improvements
from ids_xgb.matrix import DATA_FILES
from ids_xgb.pipeline import _validate_config, run_stage

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "config" / "feature_mapping.json"


def tiny_config():
    return {"stage": "E8", "sources": ["2018"], "seeds": [42], "jobs": 1,
            "group": "seven", "k_per_class": [4], "holdout_fraction": .4,
            "model_params": {"n_estimators": 4, "max_depth": 2},
            "explanations": {"enabled": False},
            "self_training": {"iterations": 2, "confidence": .6, "max_per_class": 10},
            "variants": [{"name": "ssl", "method": "self_training", "feature_engineering": True}]}


def test_cost_policy_includes_all_benign_and_ties():
    y, scores = np.array([0, 0, 1, 1]), np.array([.9, .2, .8, .7])
    m = metrics(y, scores, .5)
    assert m["total_cost"] == 100 and m["always_benign_cost"] == 2
    threshold = select_cost_threshold(y, scores)
    assert threshold == 1
    assert metrics(y, scores, threshold)["total_cost"] == 2
    # Empirical search handles duplicate scores as one indivisible decision.
    assert select_cost_threshold(np.array([0, 1]), np.array([.5, .5])) == 1
    with pytest.raises(ValueError, match="Costs"):
        metrics(y, scores, .5, fp_cost=-1)


def test_cost_threshold_matches_brute_force():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(50), np.ones(50)]
    scores = np.round(rng.uniform(size=100), 2)
    t = select_cost_threshold(y, scores, fp_cost=3, fn_cost=7)
    candidates = np.r_[1., np.unique(scores)]
    chosen = metrics(y, scores, t, fp_cost=3, fn_cost=7)["total_cost"]
    assert chosen == min(metrics(y, scores, x, fp_cost=3, fn_cost=7)["total_cost"] for x in candidates)


@pytest.mark.parametrize("filename", ["E8_improvements.json", "E8_hybrid.json", "E8_tta.json", "E8_smoke.json"])
def test_shipped_configs_validate(filename):
    config = json.loads((ROOT / "config" / filename).read_text())
    _validate_config("E8", config, MAPPING)


@pytest.mark.parametrize("change", [
    {"model_params": {"objective": "reg:squarederror"}},
    {"variants": [{"name": "../bad", "method": "weighted"}]},
    {"variants": [{"name": "bad", "method": "pooled", "adaptive_weighting": True}]},
    {"costs": {"fp": float("nan")}}, {"max_unlabeled_samples": 0},
    {"self_training": {"confidence": .4}}, {"feature_enginering": True},
])
def test_invalid_advanced_config(change):
    config = tiny_config()
    config.update(change)
    with pytest.raises(ValueError):
        validate_improvements(config)


def write_data(directory):
    directory.mkdir()
    mapping = load_mapping(MAPPING)
    rng = np.random.default_rng(3)
    for domain in ("2018", "iot"):
        frame = pd.DataFrame({spec[domain]: rng.uniform(1, 5, 100)
                              for spec in mapping["features"].values()})
        frame["Label"] = [0, 1] * 50
        frame.to_csv(directory / DATA_FILES[domain], index=False)


def test_e8_pipeline_artifacts_and_tta_roles(tmp_path):
    data_dir = tmp_path / "data"
    write_data(data_dir)
    config = tiny_config()
    config.update(threshold={"strategy": "budget_validation_cost", "calibration_fraction": .25},
                  explanations={"enabled": True, "max_samples": 4})
    config["variants"] += [
        {"name": "density", "method": "adaptive"},
        {"name": "source", "method": "source_only"},
        {"name": "target", "method": "target_only"},
        {"name": "entropy", "method": "weighted", "tta": "entropy"},
        {"name": "tta_ssl", "method": "self_training", "tta": "self_training"},
    ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    results = tmp_path / "results"
    run_id = run_stage("E8", path, data_dir, results, MAPPING)
    run = results / "runs" / run_id
    summary = pd.read_csv(run / "summary.csv")
    assert len(summary) == 6
    assert set(summary.loc[summary.tta == "none", "evaluation_role"]) == {"independent_holdout"}
    assert set(summary.loc[summary.tta != "none", "evaluation_role"]) == {"transductive_holdout"}
    assert (summary.total_cost == 100 * summary.fp + summary.fn).all()
    manifest = json.loads((run / "manifest.json").read_text())
    assert "iot_final" not in manifest["inputs_sha256"]
    assert any("predictions.npz" in key for key in manifest["artifact_sha256"])
    details = json.loads((run / "ssl_2018_s42_k4" / "run.json").read_text())
    assert details["target_labels_used"] == 8
    assert details["target_training_labels"] == 6
    assert details["target_calibration_labels"] == 2
    assert set(details["unlabeled_target_indices"]).isdisjoint(details["selected_target_indices"])
    assert (run / "ssl_2018_s42_k4" / "explanations" / "explanations.html").is_file()
    assert "total_cost_mean" in pd.read_csv(run / "aggregate.csv")


def test_evaluation_labels_do_not_influence_training_or_threshold(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    names = list(load_mapping(MAPPING)["features"])
    frame = pd.DataFrame(rng.uniform(1, 10, (80, 7)), columns=names)
    labels = np.array([0, 1] * 40)
    data = FewshotData(frame.iloc[:30], labels[:30], frame.iloc[30:60], labels[30:60],
                       frame.iloc[60:], labels[60:].copy(), np.arange(30), np.arange(30, 60), np.arange(60, 80), 42)
    monkeypatch.setattr("ids_xgb.improvements.prepare_fewshot", lambda *a, **kw: data)
    config = tiny_config()
    config["threshold"] = {"strategy": "budget_validation_cost", "calibration_fraction": .25}
    config["variants"] += [{"name": "adaptive", "method": "adaptive"},
                           {"name": "tta", "method": "weighted", "tta": "self_training"}]
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    run_improvements(config, tmp_path, MAPPING, first)
    data.y_holdout[:] = 1 - data.y_holdout
    run_improvements(config, tmp_path, MAPPING, second)
    for variant in config["variants"]:
        name = f"{variant['name']}_2018_s42_k4"
        a = np.load(first / name / "predictions.npz")
        b = np.load(second / name / "predictions.npz")
        np.testing.assert_array_equal(a["score"], b["score"])
        ra = json.loads((first / name / "run.json").read_text())
        rb = json.loads((second / name / "run.json").read_text())
        assert ra["threshold"] == rb["threshold"]


def test_e8_hybrid_smoke(tmp_path):
    pytest.importorskip("torch")
    data_dir = tmp_path / "data"
    write_data(data_dir)
    config = tiny_config()
    config["variants"] = [{"name": "hybrid", "method": "hybrid"}]
    config["hybrid"] = {"epochs": 1, "hidden_dim": 4, "embedding_dim": 2, "batch_size": 16, "max_unlabeled": 20}
    config["explanations"] = {"enabled": True, "max_samples": 4}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    run_id = run_stage("E8", path, data_dir, tmp_path / "results", MAPPING)
    output = tmp_path / "results" / "runs" / run_id / "hybrid_2018_s42_k4"
    assert (output / "hybrid_model" / "dann_encoder.pt").is_file()
    report = json.loads((output / "explanations" / "explanations.json").read_text())
    assert report["representation"] == "latent"
