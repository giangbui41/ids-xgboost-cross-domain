import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_xgb.data import impute_from_train, load_features, load_mapping, sha256
from ids_xgb.experiment import evaluate, freeze
from ids_xgb.prepare import prepare_source, prepare_iot
from ids_xgb.matrix import common_related
from ids_xgb.robustness import perturb_features
from ids_xgb.fewshot import FewshotData
from ids_xgb.pipeline import _activate, _registry, run_stage


MAPPING = Path(__file__).resolve().parents[1] / "config" / "feature_mapping.json"


def test_reading_shuffled_columns_preserves_canonical_order(tmp_path):
    mapping = load_mapping(MAPPING)
    names = mapping["groups"]["length_only"]
    frame = pd.DataFrame({mapping["features"][n]["2017"]: [i, i + 10]
                          for i, n in enumerate(names)})
    frame["Label"] = [0, 5]
    path = tmp_path / "sample.csv"
    frame[frame.columns[::-1]].to_csv(path, index=False)
    x, y = load_features(path, mapping, "length_only", "2017")
    assert list(x.columns) == names
    assert x.iloc[0].tolist() == list(range(len(names)))
    assert y.tolist() == [0, 1]


def test_missing_values_use_only_source_train():
    train = pd.DataFrame({"a": [2.0, np.nan, 4.0]})
    target = pd.DataFrame({"a": [np.nan, 100000.0]})
    x, (z,), median = impute_from_train(train, target)
    assert median["a"] == 3.0
    assert x[1, 0] == z[0, 0] == 3.0


def test_freeze_refuses_existing_destination(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "model.json").write_text("model", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"protocol": "test"}), encoding="utf-8")
    dest = tmp_path / "frozen"
    lock = freeze(run, dest)
    assert lock["model_sha256"]
    with pytest.raises(FileExistsError):
        freeze(run, dest)


def test_prepare_source_maps_labels_without_balancing(tmp_path):
    mapping = load_mapping(MAPPING)
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame({mapping["features"][n]["2017"]: np.arange(10)
                          for n in mapping["groups"]["seven"]})
    frame["Label"] = ["BENIGN"] * 8 + ["DoS"] * 2
    frame.to_csv(raw / "part.csv", index=False)
    output = tmp_path / "prepared.csv"
    report = prepare_source(raw, output, MAPPING, "2017", 1.0, 42)
    assert report["rows"] == 10
    assert pd.read_csv(output)["Label"].tolist() == [0] * 8 + [1] * 2


def test_prepare_iot_creates_disjoint_split(tmp_path):
    mapping = load_mapping(MAPPING)
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame({mapping["features"][n]["iot"]: np.arange(20) + 1
                          for n in mapping["groups"]["seven"]})
    frame["Label"] = ["Benign"] * 10 + ["DDoS"] * 10
    frame.to_csv(raw / "part.csv", index=False)
    dev, final = tmp_path / "dev.csv", tmp_path / "final.csv"
    report = prepare_iot(raw, dev, final, MAPPING, 1.0, 42)
    assert (report["dev_rows"], report["final_rows"]) == (16, 4)
    assert set(pd.read_csv(dev)["Rate"]).isdisjoint(set(pd.read_csv(final)["Rate"]))


def test_final_evaluation_requires_frozen_model_before_reading_target(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"mapping_sha256": sha256(MAPPING)}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="frozen.json"):
        evaluate(run, tmp_path / "missing_final.csv", MAPPING, "final", tmp_path / "result.json")


def test_related_features_align_by_normalized_name():
    source = pd.DataFrame({"Flow_Packets/s": [1.0], "Pkt Len Mean": [2.0]})
    target = pd.DataFrame({"Pkt Len Mean": [3.0], "Flow Packets/s": [4.0]})
    a, b, names = common_related(source, target)
    assert names == ["flow_packets_s", "pkt_len_mean"]
    assert a.iloc[0].tolist() == [1.0, 2.0]
    assert b.iloc[0].tolist() == [4.0, 3.0]


def test_drift_preserves_rate_identity_and_packet_order():
    frame = pd.DataFrame({"flow_packets_per_second": [10.0],
                          "forward_packets_per_second": [7.0],
                          "backward_packets_per_second": [3.0],
                          "packet_length_min": [2.0],
                          "packet_length_mean": [4.0],
                          "packet_length_max": [8.0],
                          "packet_length_std": [1.0]})
    for direction in ("up", "down"):
        x = perturb_features(frame, .1, direction).iloc[0]
        assert x.flow_packets_per_second == pytest.approx(
            x.forward_packets_per_second + x.backward_packets_per_second)
        assert x.packet_length_min <= x.packet_length_mean <= x.packet_length_max


def test_fewshot_budgets_are_nested_and_disjoint_from_holdout():
    target = pd.DataFrame({"f": np.arange(20)})
    data = FewshotData(pd.DataFrame({"f": [1, 2]}), np.array([0, 1]),
                       target, np.array([0] * 10 + [1] * 10),
                       pd.DataFrame({"f": [100, 101]}), np.array([0, 1]),
                       np.array([0, 1]), np.arange(20), np.array([100, 101]), 42)
    _, y_small, picked_small = data.target_sample(2)
    _, y_large, picked_large = data.target_sample(4)
    assert y_small.tolist() == [0, 0, 1, 1]
    assert set(picked_small).issubset(set(picked_large))
    assert set(picked_large).isdisjoint(data.holdout_indices)


def test_rollback_restores_original_csv_and_keeps_newer_snapshot(tmp_path):
    runs = tmp_path / "runs"
    for run_id, content in [("old", "v1\n"), ("new", "v2\n")]:
        folder = runs / run_id
        folder.mkdir(parents=True)
        (folder / "summary.csv").write_text(content)
    registry = {"runs": {r: {"stage": "E4", "summary_sha256": sha256(runs / r / "summary.csv")}
                          for r in ("old", "new")}, "active": {}, "events": []}
    _activate(tmp_path, "E4", "new", registry)
    _activate(tmp_path, "E4", "old", _registry(tmp_path))
    assert (tmp_path / "current" / "E4.csv").read_text() == "v1\n"
    assert (runs / "new" / "summary.csv").read_text() == "v2\n"
    (runs / "new" / "summary.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        _activate(tmp_path, "E4", "new", _registry(tmp_path))
    assert (tmp_path / "current" / "E4.csv").read_text() == "v1\n"


def test_final_requires_locked_selection_before_touching_results(tmp_path):
    with pytest.raises(ValueError, match="preselected source"):
        run_stage("E7", MAPPING.parent / "E7_final_evaluation.json", tmp_path / "absent-data",
                  tmp_path / "results", MAPPING)
    assert not (tmp_path / "results").exists()
