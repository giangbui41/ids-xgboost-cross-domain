import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ids_xgb.calibration import calibrate_model, calibrate_prefit, calibration_metrics, check_calibration
from ids_xgb.data import load_features, load_mapping
from ids_xgb.fewshot import improvement_config
from ids_xgb.matrix import DATA_FILES
from ids_xgb.features import FeatureEngineer
from ids_xgb.pipeline import _validate_config, run_stage

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "config/feature_mapping.json"


def test_calibration_metrics_known_values_and_endpoints():
    report = calibration_metrics([0, 1], [0., 1.])
    assert report["ece"] == report["brier_score"] == 0
    report = calibration_metrics([0, 1], [.75, .75])
    assert report["ece"] == .25
    assert report["brier_score"] == .3125
    assert sum(row["count"] for row in report["reliability"]) == 2
    with pytest.raises(ValueError):
        calibration_metrics([0, 1], [np.nan, 1])


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_prefit_calibration_preserves_base_and_serializes(tmp_path, method):
    x = np.arange(80).reshape(-1, 1)
    y = (np.arange(80) % 3 == 0).astype(int)
    base = LogisticRegression().fit(x[:40], y[:40])
    original = base.predict_proba(x).copy()
    calibrated = calibrate_prefit(base, x[40:60], y[40:60], method=method)
    assert np.array_equal(base.predict_proba(x), original)
    path = tmp_path / "model.joblib"
    joblib.dump(calibrated, path)
    assert np.allclose(joblib.load(path).predict_proba(x[60:]), calibrated.predict_proba(x[60:]))
    assert check_calibration(calibrated, x[60:], y[60:])["n"] == 20
    cv_model = calibrate_model(base, x[:60], y[:60], cv=3, method=method)
    assert cv_model.predict_proba(x[60:]).shape == (20, 2)


@pytest.mark.parametrize("filename", ["E4_fewshot.json", "E4_fewshot_smoke.json", "E4_tta.json"])
def test_e4_configs_validate_without_hybrid(filename):
    config = json.loads((ROOT / "config" / filename).read_text())
    _validate_config("E4", config, MAPPING)
    assert all(v["method"] != "hybrid" for v in improvement_config(config)["variants"])


def test_e4_calibration_budget_predictions_and_holdout_isolation(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    mapping = load_mapping(MAPPING)
    rng = np.random.default_rng(5)
    for domain in ("2018", "iot"):
        frame = pd.DataFrame({spec[domain]: rng.uniform(1, 10, 120) for spec in mapping["features"].values()})
        frame["Label"] = np.tile([0, 1], 60)
        frame.to_csv(data_dir / DATA_FILES[domain], index=False)
    config = json.loads((ROOT / "config/E4_fewshot_smoke.json").read_text())
    config.update(k_per_class=[4], jobs=1, use_tta=True, tta_method="self_training")
    config["model_params"]["n_estimators"] = 3
    config["calibration"]["fraction"] = .25
    config["explanations"]["max_samples"] = 3
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    original = pd.read_csv

    def no_final(path, *args, **kwargs):
        assert "final" not in str(path).lower()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", no_final)
    run_id = run_stage("E4", path, data_dir, tmp_path / "results", MAPPING)
    run = tmp_path / "results/runs" / run_id
    summary = pd.read_csv(run / "summary.csv")
    assert set(summary.stage) == {"E4"}
    assert len(summary) == 9
    folder = run / "weighted_calibrated_2018_s42_k4"
    details = json.loads((folder / "run.json").read_text())
    split = json.loads((run / "split_2018_s42.json").read_text())
    picked = set(details["selected_target_indices"])
    calibration = set(details["calibration_target_indices"])
    assert len(picked) == 8 and len(calibration) == 2
    assert details["target_training_labels"] == 6
    assert calibration <= picked
    assert picked.isdisjoint(details["unlabeled_target_indices"])
    assert picked.isdisjoint(split["holdout_indices"])
    assert set(details["unlabeled_target_indices"]).isdisjoint(split["holdout_indices"])
    x, _ = load_features(data_dir / DATA_FILES["iot"], mapping, "seven", "iot")
    engineered = FeatureEngineer().fit_transform(x.iloc[split["holdout_indices"]])
    matrix = engineered.fillna(pd.Series(details["medians"])).to_numpy(np.float32)
    restored = joblib.load(folder / "calibrated_model.joblib")
    predictions = np.load(folder / "predictions.npz")
    assert np.allclose(restored.predict_proba(matrix)[:, 1], predictions["score"])
    assert (folder / "calibration.json").exists()
    assert (folder / "explanations/explanations.html").exists()
    tta = summary[summary.tta == "self_training"].iloc[0]
    assert tta.evaluation_role == "transductive_holdout"
    assert (summary[summary.tta == "none"].evaluation_role == "independent_holdout").all()
