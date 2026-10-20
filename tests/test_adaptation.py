import numpy as np
import pytest

from ids_xgb.adaptation import adaptive_source_weights, entropy_temperature, self_train


def test_density_favors_target_region_with_bounded_mean_one_weights():
    rng = np.random.default_rng(17)
    source = rng.normal(size=(500, 2))
    target = rng.normal(loc=1.5, size=(100, 2))
    weights, report = adaptive_source_weights(source, (source[:, 0] > 0).astype(int),
                                               np.empty((0, 2)), np.empty(0), target,
                                               min_weight=.2, max_weight=3.)
    assert weights.min() >= .2
    assert weights.max() <= 3
    assert weights.mean() == pytest.approx(1.)
    assert weights[source[:, 0] > 1].mean() > weights[source[:, 0] < -1].mean()
    assert report["conflict_assessed"] is False
    assert report["conflict_fraction"] is None
    assert report["scaler_fit_domain"] == "source"


def test_conflict_diagnostic_uses_only_the_labeled_target_budget():
    rng = np.random.default_rng(3)
    source = np.concatenate([rng.normal(-5, .2, size=(50, 1)), rng.normal(5, .2, size=(50, 1))])
    source_y = np.repeat([0, 1], 50)
    target_y = 1 - source_y
    _, report = adaptive_source_weights(source, source_y, source, target_y, source)
    assert report["conflict_assessed"] is True
    assert report["conflict_fraction"] > .9
    assert "heuristic" in report["conflict_interpretation"]


def test_uncertainty_is_reproducible_and_handles_missing_target_classes():
    source = np.arange(20.).reshape(-1, 1)
    weights, report = adaptive_source_weights(source, np.tile([0, 1], 10),
                                               source[:2], [0, 0], source[2:], strategy="uncertainty")
    np.testing.assert_array_equal(weights, np.ones(20))
    assert report["status"] == "uniform_insufficient_target_classes"
    first, _ = adaptive_source_weights(source, np.tile([0, 1], 10), source, (source[:, 0] > 10).astype(int),
                                        source, strategy="uncertainty")
    second, _ = adaptive_source_weights(source, np.tile([0, 1], 10), source, (source[:, 0] > 10).astype(int),
                                         source, strategy="uncertainty")
    np.testing.assert_array_equal(first, second)


class RecordingModel:
    classes_ = np.array([0, 1])

    def __init__(self, fits, confident=True):
        self.fits = fits
        self.confident = confident

    def fit(self, x, y, sample_weight):
        self.fits.append((x.copy(), y.copy(), sample_weight.copy()))
        return self

    def predict_proba(self, x):
        score = np.where(x[:, 0] < 0, .01, .99) if self.confident else np.full(len(x), .5)
        return np.column_stack([1 - score, score])


def test_self_training_never_reselects_rows_and_caps_cumulative_pseudo_mass():
    fits = []
    x, y, w = np.array([[-100.], [100.]]), np.array([0, 1]), np.array([2., 3.])
    pool = np.concatenate([np.arange(-10., 0.), np.arange(1., 11.)]).reshape(-1, 1)
    _, history = self_train(lambda: RecordingModel(fits), x, y, w, pool, iterations=5,
                             max_per_class=3, pseudo_weight=.2)
    selected = [i for entry in history for i in entry["selected_indices"]]
    assert len(selected) == len(set(selected)) == len(pool)
    assert len(fits) == 5  # Initial fit plus four rounds, then the pool is exhausted.
    for train_x, train_y, train_w in fits:
        np.testing.assert_array_equal(train_x[:2], x)
        np.testing.assert_array_equal(train_y[:2], y)
        np.testing.assert_array_equal(train_w[:2], w)
        assert train_w[2:].sum() <= .2 * w.sum() + 1e-12
        assert len(np.unique(train_x[:, 0])) == len(train_x)
    assert history[-1]["stop_reason"] == "pool_exhausted"
    assert all(entry["selected_class_counts"]["0"] <= 3 for entry in history)
    repeated_fits = []
    _, repeated = self_train(lambda: RecordingModel(repeated_fits), x, y, w, pool, iterations=5,
                              max_per_class=3, pseudo_weight=.2)
    assert repeated == history


def test_self_training_empty_and_unconfident_pools_need_only_one_fit():
    x, y = np.array([[-1.], [1.]]), np.array([0, 1])
    fits = []
    _, history = self_train(lambda: RecordingModel(fits), x, y, None, np.empty((0, 1)))
    assert len(fits) == 1 and history == []
    fits = []
    _, history = self_train(lambda: RecordingModel(fits, confident=False), x, y, None, x)
    assert len(fits) == 1
    assert history[0]["selected_count"] == 0
    assert history[0]["stop_reason"] == "no_confident_predictions"


def test_entropy_temperature_is_finite_bounded_and_preserves_decision_order():
    scores = np.array([0., .01, .2, .49, .5, .51, .8, .99, 1.])
    adapted, report = entropy_temperature(scores)
    assert np.isfinite(adapted).all()
    assert np.all((adapted > 0) & (adapted < 1))
    assert np.all(np.diff(adapted) >= 0)
    assert report["temperature"] >= .5 and report["temperature"] <= 2
    assert report["entropy_after"] <= report["entropy_before"]
    assert report["labels_used"] == 0
    np.testing.assert_array_equal(adapted >= .5, scores >= .5)
    empty, empty_report = entropy_temperature(np.empty(0))
    assert len(empty) == 0 and empty_report["entropy_after"] == 0


@pytest.mark.parametrize("kwargs", [{"confidence": .5}, {"iterations": -1}, {"max_per_class": 0},
                                   {"pseudo_weight": 0}, {"pseudo_weight": 2}])
def test_self_training_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        self_train(lambda: RecordingModel([]), [[0.], [1.]], [0, 1], [1., 1.], [[2.]], **kwargs)


def test_adaptation_rejects_unimputed_data_and_impossible_bounds():
    with pytest.raises(ValueError, match="finite"):
        adaptive_source_weights([[np.nan]], [0], [[1.]], [1], [[2.]])
    with pytest.raises(ValueError, match="bounds"):
        adaptive_source_weights([[0.]], [0], [[1.]], [1], [[2.]], min_weight=2.)
    with pytest.raises(ValueError):
        entropy_temperature([1.1])
    with pytest.raises(ValueError):
        entropy_temperature([.3], min_temperature=0.)
