import builtins
import json

import numpy as np
import pytest

from ids_xgb.hybrid import HybridDANNXGB, _require_torch, _reverse_gradient, _weighted_task_loss


def _data():
    rng = np.random.default_rng(123)
    labels = np.tile([0, 1], 32)
    source = rng.normal(0, .25, size=(len(labels), 3)).astype(np.float32)
    source[:, 0] += np.where(labels == 1, 2, -2)
    target = source[:4].copy()
    target[:, 2] += 8
    unlabeled = source[4:44].copy()
    unlabeled[:, 2] += 8
    return source, labels, target, labels[:4], unlabeled


def test_missing_torch_has_actionable_optional_dependency_error(monkeypatch):
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(ImportError, match="optional PyTorch.*hybrid"):
        _require_torch()


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"batch_size": True}, {"max_unlabeled": 0},
    {"lambda_domain": -1}, {"learning_rate": float("nan")},
    {"xgb_params": {"objective": "multi:softmax"}},
    {"xgb_params": {"early_stopping_rounds": 5}},
])
def test_invalid_config_fails_without_torch(kwargs):
    with pytest.raises(ValueError):
        HybridDANNXGB(**kwargs)


def test_validation_happens_before_optional_torch_import(monkeypatch):
    import ids_xgb.hybrid as hybrid

    def should_not_import():
        pytest.fail("Input validation must run before importing torch")

    monkeypatch.setattr(hybrid, "_require_torch", should_not_import)
    xs, ys, xt, yt, xu = _data()
    xs[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HybridDANNXGB().fit(xs, ys, xt, yt, xu)
    xs[0, 0] = 0
    with pytest.raises(ValueError, match="features"):
        HybridDANNXGB().fit(xs, ys, xt[:, :2], yt, xu)
    with pytest.raises(ValueError, match="binary label"):
        HybridDANNXGB().fit(xs, ys, xt, [0, 1, 7, 1], xu)
    with pytest.raises(ValueError, match="sample_weight"):
        HybridDANNXGB().fit(xs, ys, xt, yt, xu, sample_weight=np.ones(len(xs)))
    with pytest.raises(ValueError, match="target adaptation rows"):
        HybridDANNXGB().fit(xs, ys, xs[:0], ys[:0], xs[:0])


def test_gradient_reversal_preserves_forward_and_reverses_backward():
    torch = pytest.importorskip("torch")
    value = torch.tensor([1., -2.], requires_grad=True)
    reversed_value = _reverse_gradient(value, .3)
    assert torch.equal(value, reversed_value)
    reversed_value.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, -.3))


@pytest.mark.parametrize("shuffle", [False, True])
def test_scarce_target_weight_mass_survives_minibatches_and_short_final_batch(shuffle):
    torch = pytest.importorskip("torch")
    # Three target labels carry 60% of the task mass. Without shuffling, all
    # three fall in the short final batch and must retain precisely that mass.
    n_source, n_target, batch_size = 101, 3, 16
    count = n_source + n_target
    weight = np.concatenate([np.full(n_source, .4 / n_source),
                             np.full(n_target, .6 / n_target)])
    weight = torch.tensor(weight / weight.mean(), dtype=torch.float64)
    source_logit = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    target_logit = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    order = np.random.default_rng(42).permutation(count) if shuffle else np.arange(count)
    for start in range(0, count, batch_size):
        index = torch.from_numpy(order[start:start + batch_size])
        logits = torch.cat([source_logit.expand(n_source), target_logit.expand(n_target)])[index]
        loss = _weighted_task_loss(torch, logits, torch.zeros_like(logits), weight[index], batch_size)
        loss.backward()
    assert float(target_logit.grad / source_logit.grad) == pytest.approx(.6 / .4)


def test_hybrid_source_scaling_determinism_and_artifact_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    xs, ys, xt, yt, xu = _data()
    source_before, unlabeled_before = xs.copy(), xu.copy()
    params = dict(seed=42, jobs=1, hidden_dim=8, embedding_dim=4,
                  epochs=3, batch_size=16, max_unlabeled=12,
                  xgb_params={"n_estimators": 20, "max_depth": 2, "learning_rate": .2})
    weights = np.concatenate([np.ones(len(xs)), np.full(len(xt), 2)])
    model = HybridDANNXGB(**params).fit(xs, ys, xt, yt, xu, sample_weight=weights)
    np.testing.assert_allclose(model.scaler_.mean_, xs.mean(axis=0, dtype=np.float64))
    np.testing.assert_array_equal(xs, source_before)
    np.testing.assert_array_equal(xu, unlabeled_before)
    assert model.n_unlabeled_used_ == 12
    assert model.transform(xs).shape == (len(xs), 4)
    assert len(model.history_) == 3
    assert all(np.isfinite(row["task_loss"]) and np.isfinite(row["domain_loss"])
               for row in model.history_)
    assert all(not parameter.requires_grad for parameter in model.encoder_.parameters())
    probabilities = model.predict_proba(xs)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert np.mean((probabilities[:, 1] >= .5) == ys) >= .85
    assert model.predict_proba(xs[:0]).shape == (0, 2)
    again = HybridDANNXGB(**params).fit(xs, ys, xt, yt, xu, sample_weight=weights)
    np.testing.assert_array_equal(again.transform(xs), model.transform(xs))
    np.testing.assert_array_equal(again.predict_proba(xs), probabilities)
    assert again.history_ == model.history_

    destination = tmp_path / "hybrid"
    model.save(destination)
    saved = json.loads((destination / "hybrid.json").read_text(encoding="utf-8"))
    assert saved["n_unlabeled_used"] == 12
    restored = HybridDANNXGB.load(destination)
    np.testing.assert_array_equal(restored.transform(xs), model.transform(xs))
    np.testing.assert_array_equal(restored.predict_proba(xs), probabilities)
    assert restored.embedding_feature_names_ == model.embedding_feature_names_
    assert restored.history_ == model.history_
    with pytest.raises(FileExistsError):
        model.save(destination)
    # Serialization is a tensor-only state dict, not a custom model pickle.
    state = torch.load(destination / "dann_encoder.pt", weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in state.values())


def test_hybrid_supports_zero_target_labels_and_domain_gradient_changes_encoder():
    pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    xs, ys, _, _, xu = _data()
    params = dict(jobs=1, epochs=2, batch_size=16, hidden_dim=8, embedding_dim=4,
                  xgb_params={"n_estimators": 3, "max_depth": 2})
    adaptive = HybridDANNXGB(**params, lambda_domain=1).fit(xs, ys, xs[:0], ys[:0], xu)
    plain = HybridDANNXGB(**params, lambda_domain=0).fit(xs, ys, xs[:0], ys[:0], xu)
    assert np.max(np.abs(adaptive.transform(xs) - plain.transform(xs))) > 1e-5
    assert adaptive.predict_proba(xu).shape == (len(xu), 2)
