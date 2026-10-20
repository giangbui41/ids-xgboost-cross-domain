import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from ids_xgb.features import CANONICAL_FEATURES, FeatureEngineer, causal_temporal_features
from ids_xgb.explanations import export_tree_explanations


def seven_frame():
    return pd.DataFrame([[10., 7., 3., 2., 4., 8., 1.],
                         [0., 0., 0., 0., 0., 0., 0.]], columns=CANONICAL_FEATURES)


def test_engineering_has_no_population_or_target_statistics():
    source = seven_frame()
    target = source * 100
    engineer = FeatureEngineer().fit(source)
    expected = engineer.transform(target)
    target.iloc[1] = 1e10
    pd.testing.assert_series_equal(engineer.transform(target).iloc[0], expected.iloc[0])
    # Fit learns schema only: changing every fitted value cannot alter transform.
    pd.testing.assert_frame_equal(FeatureEngineer().fit(source * 1e6).transform(target),
                                  engineer.transform(target))
    assert engineer.feature_names_in_ == list(source.columns)
    assert engineer.feature_names_out_ == list(expected.columns)
    assert expected.loc[0, "eng_forward_fraction"] == pytest.approx(.7)
    assert expected.loc[0, "eng_length_coefficient_of_variation"] == pytest.approx(.25)
    assert expected.loc[0, "eng_rate_length_interaction"] == pytest.approx(400000.)


def test_engineering_preserves_originals_handles_zero_and_undefined_ratios():
    source = seven_frame()
    result = FeatureEngineer().fit_transform(source)
    pd.testing.assert_frame_equal(result.loc[:, source.columns], source)
    assert np.isfinite(result.iloc[1].to_numpy()).all()
    assert (result.iloc[1] == 0).all()
    invalid = source.copy()
    invalid.loc[0, "packet_length_mean"] = 0
    invalid.loc[0, "flow_packets_per_second"] = np.inf
    invalid.loc[0, "packet_length_max"] = 1e100
    transformed = FeatureEngineer().fit_transform(invalid)
    assert not np.isinf(transformed.to_numpy()).any()
    assert np.isnan(transformed.loc[0, "eng_length_coefficient_of_variation"])
    assert np.isnan(transformed.loc[0, "flow_packets_per_second"])


def test_engineering_supports_feature_ablation_and_disabled_mode():
    lengths = seven_frame().filter(like="packet_length")
    expanded = FeatureEngineer().fit_transform(lengths)
    assert "eng_length_span" in expanded
    assert "eng_forward_fraction" not in expanded
    pd.testing.assert_frame_equal(FeatureEngineer(enabled=False).fit_transform(lengths), lengths)
    with pytest.raises(ValueError, match="schema"):
        FeatureEngineer().fit(lengths).transform(lengths[lengths.columns[::-1]])
    with pytest.raises(ValueError, match="Label"):
        FeatureEngineer().fit(lengths.assign(Label=[0, 1]))
    with pytest.raises(ValueError, match="metadata"):
        FeatureEngineer(temporal={"window": 5})


def test_temporal_features_are_prior_only_grouped_and_preserve_order():
    frame = pd.DataFrame({"rate": [30., 100., 10., 20., 40.]}, index=[9, 7, 5, 3, 1])
    metadata = pd.DataFrame({"device": ["a", "b", "a", "a", "a"],
                             "time": [3., 1., 1., 2., 4.]}, index=frame.index)
    output = causal_temporal_features(frame, metadata, "time", ["device"], window=2)
    assert output.index.tolist() == frame.index.tolist()
    assert np.isnan(output.loc[5, "temporal_prior_mean_rate"])
    assert np.isnan(output.loc[7, "temporal_prior_mean_rate"])
    assert output.loc[9, "temporal_prior_mean_rate"] == pytest.approx(15.)
    assert output.loc[1, "temporal_prior_mean_rate"] == pytest.approx(25.)
    assert output.loc[1, "temporal_prior_std_rate"] == pytest.approx(5.)
    assert output.loc[9, "temporal_prior_count"] == 2
    assert output.loc[9, "temporal_gap_seconds"] == 1
    changed = frame.copy()
    changed.loc[9, "rate"] = 9000  # Current event and all later events must have no effect.
    changed.loc[1, "rate"] = 8000
    transformed = causal_temporal_features(changed, metadata, "time", ["device"], window=2)
    pd.testing.assert_series_equal(transformed.loc[9].filter(like="temporal_"),
                                    output.loc[9].filter(like="temporal_"))


def test_temporal_rejects_missing_metadata_ties_and_misalignment():
    frame = pd.DataFrame({"rate": [10., 20.]})
    metadata = pd.DataFrame({"device": ["a", "a"], "time": [1., 1.]})
    with pytest.raises(ValueError, match="ambiguous"):
        causal_temporal_features(frame, metadata, "time", ["device"])
    with pytest.raises(ValueError, match="aligned"):
        causal_temporal_features(frame, metadata.iloc[::-1], "time", ["device"])
    with pytest.raises(ValueError, match="metadata"):
        causal_temporal_features(frame, metadata.drop(columns="time"), "time", ["device"])
    with pytest.raises(ValueError, match="group columns"):
        causal_temporal_features(frame, metadata, "time", [])


def test_temporal_string_timestamps_have_seconds_gap():
    frame = pd.DataFrame({"rate": [10., 20.]})
    metadata = pd.DataFrame({"device": ["a", "a"],
                             "time": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:30Z"]})
    output = causal_temporal_features(frame, metadata, "time", ["device"])
    assert output.loc[1, "temporal_gap_seconds"] == pytest.approx(30.)
    assert output.loc[1, "temporal_prior_mean_rate"] == pytest.approx(10.)


@pytest.fixture
def synthetic_model():
    rng = np.random.default_rng(10)
    source = rng.normal(size=(90, 3)).astype(np.float32)
    target = rng.normal(loc=[1., 0., -.5], size=(75, 3)).astype(np.float32)
    labels = (source[:, 0] + .5 * source[:, 1] > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=2, learning_rate=.2, n_jobs=1,
                          random_state=1, tree_method="hist")
    model.fit(source, labels)
    return model, source, target


def test_native_tree_shap_additivity_scores_artifacts_and_bounded_sampling(tmp_path, synthetic_model):
    model, source, target = synthetic_model
    names = ["rate", "length", "variation"]
    summary = export_tree_explanations(model, source, target, names, tmp_path, max_samples=13)
    assert summary["source_samples"] == summary["target_samples"] == 13
    for domain, data in [("source", source), ("target", target)]:
        frame = pd.read_csv(tmp_path / f"{domain}_contributions.csv")
        reconstructed = frame[["shap__" + n for n in names]].sum(axis=1) + frame.bias
        assert np.allclose(reconstructed, frame.raw_margin, atol=1e-5)
        expected = model.predict_proba(data[frame.row_position])[:, 1]
        assert np.allclose(expected, frame.probability_attack, atol=1e-6)
    comparison = pd.read_csv(tmp_path / "domain_comparison.csv")
    assert comparison.feature.tolist() == names
    assert "delta_mean_signed_target_minus_source" in comparison
    html = (tmp_path / "explanations.html").read_text(encoding="utf-8")
    assert "Individual decision" in html and "<svg" in html
    assert "causal bias" in html and "<script src=" not in html
    assert json.loads((tmp_path / "explanations.json").read_text())["method"] == "native_tree_shap"


@pytest.mark.parametrize("prefix", ["latent_", "dann_embedding_"])
def test_tree_shap_domain_samples_independent_reproducible_and_latent_labeled(tmp_path, synthetic_model, prefix):
    model, source, target = synthetic_model
    names = [prefix + str(i) for i in range(3)]
    first = export_tree_explanations(model, source, target, names, tmp_path / "first", max_samples=7)
    second = export_tree_explanations(model, source[:20], target, names, tmp_path / "second", max_samples=7)
    third = export_tree_explanations(model, source, target, names, tmp_path / "third", max_samples=7)
    assert first["representation"] == "latent"
    assert first["target_row_positions"] == second["target_row_positions"]
    assert first["source_row_positions"] == third["source_row_positions"]
    assert first["target_row_positions"] == third["target_row_positions"]


def test_tree_shap_artifact_links_survive_snapshot_move(tmp_path, synthetic_model):
    model, source, target = synthetic_model
    staging = tmp_path / "run.running"
    snapshot = tmp_path / "run_complete"
    summary = export_tree_explanations(model, source, target, ["a", "b", "c"], staging, max_samples=3)
    staging.rename(snapshot)
    saved = json.loads((snapshot / "explanations.json").read_text())
    assert saved["files"] == summary["files"]
    for filename in saved["files"].values():
        assert "/" not in filename and "\\" not in filename
        assert (snapshot / filename).is_file()


def test_tree_shap_respects_best_iteration(tmp_path, synthetic_model):
    model, source, target = synthetic_model
    model.get_booster().set_attr(best_iteration="2")
    summary = export_tree_explanations(model, source, target, ["a", "b", "c"], tmp_path, max_samples=5)
    assert summary["iteration_range"] == [0, 3]
    frame = pd.read_csv(tmp_path / "target_contributions.csv")
    assert np.allclose(frame.probability_attack,
                       model.predict_proba(target[frame.row_position])[:, 1], atol=1e-6)


def test_tree_shap_rejects_mismatched_names_and_non_trees(tmp_path, synthetic_model):
    model, source, target = synthetic_model
    with pytest.raises(ValueError, match="feature count"):
        export_tree_explanations(model, source, target, ["a"], tmp_path)
    with pytest.raises(ValueError, match="XGBoost"):
        export_tree_explanations(object(), source, target, ["a", "b", "c"], tmp_path)
    model.get_booster().feature_names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="names/order"):
        export_tree_explanations(model, source, target, ["b", "a", "c"], tmp_path)
