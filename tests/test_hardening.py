"""Regression checks; all datasets and FINAL markers live in pytest temp dirs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_xgb.data import claim_final_evaluation, load_mapping, overlap_report
from ids_xgb.experiment import metrics
from ids_xgb.matrix import DATA_FILES, common_related
from ids_xgb.pipeline import (
    STAGE_CONFIGS, _exclusive_registry, _input_hashes,
    _verify_final_lock, lock_final, run_stage,
)
from ids_xgb.prepare import binary_label, prepare_iot


MAPPING = Path(__file__).resolve().parents[1] / "config" / "feature_mapping.json"


def test_related_mapping_aligns_abbreviations_in_both_directions():
    mapping = load_mapping(MAPPING)
    frames = {domain: pd.DataFrame({spec[domain]: [i, i + 1]
                                   for i, spec in enumerate(mapping["features"].values())})
              for domain in ("2017", "2018")}
    for source, target in [("2017", "2018"), ("2018", "2017")]:
        a, b, names = common_related(frames[source], frames[target], mapping, source, target)
        assert set(names) == set(mapping["features"])
        pd.testing.assert_frame_equal(a, b)


def test_related_mapping_rejects_alias_collision():
    mapping = load_mapping(MAPPING)
    frame = pd.DataFrame({"Flow_Packets/s": [1], "flow_packets_per_second": [2]})
    with pytest.raises(ValueError, match="Ambiguous"):
        common_related(frame, frame, mapping)


def test_hashes_require_only_selected_inputs(tmp_path):
    for domain in ("2018", "iot"):
        (tmp_path / DATA_FILES[domain]).write_text(domain)
    hashes = _input_hashes(tmp_path, "E4", MAPPING, {"sources": ["2018"]})
    assert set(hashes) == {"2018", "iot", "feature_mapping"}
    hashes = _input_hashes(tmp_path, "E1", MAPPING, {"scenarios": ["E1_IoT"]})
    assert set(hashes) == {"iot", "feature_mapping"}


def test_overlap_audit_ignores_index_and_normalizes_numeric_types():
    a = pd.DataFrame({"a": [1, 1, 2], "b": [4, 4, 5]}, index=[10, 20, 30])
    b = pd.DataFrame({"a": [1.0, 3.0], "b": [4.0, 6.0]})
    report = overlap_report(a, b)
    assert report["left_duplicate_vectors"] == 1
    assert report["left_rows_matching_right"] == 2
    assert report["right_rows_matching_left"] == 1


def test_mixed_numeric_and_text_labels():
    assert binary_label(pd.Series(["0.0", "BENIGN", "DoS", "2"])).tolist() == [0, 0, 1, 1]
    for bad in ("-1", "inf", "nan"):
        with pytest.raises(ValueError):
            binary_label(pd.Series(["Benign", bad]))


def test_prepare_refuses_same_dev_final_path(tmp_path):
    with pytest.raises(ValueError, match="paths must differ"):
        prepare_iot(tmp_path, tmp_path / "same.csv", tmp_path / "same.csv", MAPPING, 1, 42)


def test_prepare_deduplication_and_conflicting_labels(tmp_path):
    mapping = load_mapping(MAPPING)
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame({spec["iot"]: np.arange(20, dtype=float)
                          for spec in mapping["features"].values()})
    frame["Label"] = ["Benign"] * 10 + ["Attack"] * 10
    pd.concat([frame, frame]).to_csv(raw / "part.csv", index=False)
    report = prepare_iot(raw, tmp_path / "dev.csv", tmp_path / "final.csv", MAPPING,
                         1, 42, deduplicate=True)
    assert report["rows_removed"] == 20
    dev = pd.read_csv(tmp_path / "dev.csv").drop(columns="Label")
    final = pd.read_csv(tmp_path / "final.csv").drop(columns="Label")
    assert overlap_report(dev, final)["left_rows_matching_right"] == 0
    conflict = frame.iloc[[0]].copy()
    conflict["Label"] = "Attack"
    pd.concat([frame, conflict]).to_csv(raw / "part.csv", index=False)
    with pytest.raises(ValueError, match="conflicting labels"):
        prepare_iot(raw, tmp_path / "dev2.csv", tmp_path / "final2.csv", MAPPING,
                    1, 42, deduplicate=True)


def test_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError, match="both binary classes"):
        metrics(np.array([1, 1]), np.array([.2, .8]), .5)
    with pytest.raises(ValueError, match="finite probabilities"):
        metrics(np.array([0, 1]), np.array([np.nan, .8]), .5)


def final_config():
    return {"stage": "E7", "source": "2018", "method": "target_only",
            "k_per_class": 2, "seed": 42, "model": "logistic_regression"}


def test_final_lock_does_not_require_final_and_detects_changes(tmp_path):
    for domain in ("2018", "iot"):
        (tmp_path / DATA_FILES[domain]).write_text(domain)
    path = tmp_path / "selected.json"
    config = final_config()
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="lock-final"):
        run_stage("E7", path, tmp_path, tmp_path / "results", MAPPING)
    assert not (tmp_path / "results").exists()
    lock_final(path, tmp_path, MAPPING, "Predeclared choice")
    _verify_final_lock(path, tmp_path, MAPPING, config)
    with pytest.raises(FileExistsError):
        lock_final(path, tmp_path, MAPPING, "Second choice")
    config["k_per_class"] = 3
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="changed"):
        _verify_final_lock(path, tmp_path, MAPPING, config)


def test_final_attempt_is_shared_across_result_paths(tmp_path):
    target = tmp_path / "final.csv"
    target.write_text("fixture only")
    claim_final_evaluation(target, {"output": "first/results"})
    with pytest.raises(ValueError, match="already attempted"):
        claim_final_evaluation(target, {"output": "second/results"})


def test_registry_lock_rejects_concurrent_writer_and_releases_on_error(tmp_path):
    with pytest.raises(RuntimeError, match="busy"):
        with _exclusive_registry(tmp_path):
            with _exclusive_registry(tmp_path):
                pass
    assert not (tmp_path / ".registry.lock").exists()


@pytest.mark.parametrize("stage", [f"E{i}" for i in range(7)])
def test_stage_end_to_end_on_synthetic_data(tmp_path, stage):
    mapping = load_mapping(MAPPING)
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(42)
    for domain in ("2017", "2018", "iot"):
        frame = pd.DataFrame({spec[domain]: rng.uniform(1, 10, 80)
                              for spec in mapping["features"].values()})
        frame["Label"] = [0, 1] * 40
        frame.to_csv(data / DATA_FILES[domain], index=False)
    config = json.loads((MAPPING.parent / STAGE_CONFIGS[stage]).read_text(encoding="utf-8"))
    config.update(seeds=[42], models=["logistic_regression"], model="logistic_regression",
                  sources=["2018"], jobs=1)
    if stage == "E4":
        # Keep regression coverage for legacy E4/LogisticRegression configs.
        # Improved XGBoost E4 is covered in test_calibration_integration.py.
        legacy_keys = {"stage", "description", "sources", "group", "model", "seeds",
                       "holdout_fraction", "k_per_class", "methods", "alpha", "jobs"}
        config = {key: value for key, value in config.items() if key in legacy_keys}
        config["methods"] = ["source_only", "target_only", "pooled", "weighted"]
        config["k_per_class"] = [2]
    elif stage in {"E5", "E6"}:
        config["k_per_class"] = 2
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    results = tmp_path / "results"
    run_id = run_stage(stage, path, data, results, MAPPING)
    assert not pd.read_csv(results / "current" / f"{stage}.csv").empty
    manifest = json.loads((results / "runs" / run_id / "manifest.json").read_text())
    assert "iot_final" not in manifest["inputs_sha256"]
    if stage == "E4":
        selected = final_config()
        selected_path = tmp_path / "selected.json"
        selected_path.write_text(json.dumps(selected))
        # A synthetic FINAL, never the project's real evaluation dataset.
        final_frame = frame.copy()
        final_frame.iloc[:, :-1] += .25
        final_frame.to_csv(data / DATA_FILES["iot_final"], index=False)
        lock_final(selected_path, data, MAPPING, "Synthetic integration test")
        run_stage("E7", selected_path, data, results, MAPPING)
        with pytest.raises(ValueError, match="already attempted"):
            run_stage("E7", selected_path, data, tmp_path / "other_results", MAPPING)


@pytest.mark.parametrize("change", [{"seeds": []}, {"methods": ["typo"]},
                                  {"k_per_class": [0]}, {"alpha": 1}, {"holdout_fraction": 0},
                                  {"seeds": [42, 42]}, {"k_per_class": None}, {"alpha": None}])
def test_invalid_config_fails_before_creating_results(tmp_path, change):
    config = json.loads((MAPPING.parent / STAGE_CONFIGS["E4"]).read_text())
    config.update(change)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run_stage("E4", path, tmp_path / "missing", tmp_path / "results", MAPPING)
    assert not (tmp_path / "results").exists()
