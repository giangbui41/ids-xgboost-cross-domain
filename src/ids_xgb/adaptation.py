"""Adaptation primitives whose target inputs have no evaluation labels.

All feature matrices must already be aligned and imputed using training data.
The caller owns the split: pass adaptation-pool features to these functions.
Passing evaluation features is a separate, explicitly transductive experiment.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _features(values, name: str, width: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if width is not None and result.shape[1] != width:
        raise ValueError(f"{name} has a different number of features")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite, training-imputed features")
    return result


def _labels(values, count: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != count or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must contain one binary label per feature row")
    return result.astype(np.int64, copy=False)


def _bounded_normalize(raw: np.ndarray, low: float, high: float) -> np.ndarray:
    """Preserve ranking, exact bounds, and mean one simultaneously."""
    if low == high == 1:
        return np.ones_like(raw)
    left, right = 0.0, high / float(raw.min())
    for _ in range(80):
        scale = (left + right) / 2
        if np.clip(raw * scale, low, high).mean() < 1:
            left = scale
        else:
            right = scale
    return np.clip(raw * ((left + right) / 2), low, high)


def adaptive_source_weights(xs, ys, xt_labeled, yt_labeled, xu, *, seed: int = 42,
                            strategy: str = "density", min_weight: float = .1,
                            max_weight: float = 10.0):
    """Return bounded mean-one source multipliers and diagnostic measurements.

    Density ratios use a balanced source/target logistic domain discriminator.
    Its scaler sees source features only; its target covariates come from ``xu``
    (or the labeled adaptation budget when that pool is empty). Each domain is
    deterministically subsampled to at most 20,000 fitting rows.

    Uncertainty weighting favors source rows a target-budget classifier finds
    ambiguous. Both strategies attenuate confidently contradictory source labels
    when the target budget includes both classes. This is a noisy heuristic,
    particularly at small label budgets, and does not prove domain conflict.
    """
    if strategy not in {"density", "uncertainty"}:
        raise ValueError("strategy must be 'density' or 'uncertainty'")
    if not (np.isfinite(min_weight) and np.isfinite(max_weight)
            and 0 < min_weight <= 1 <= max_weight):
        raise ValueError("weight bounds must satisfy 0 < min_weight <= 1 <= max_weight")
    source = _features(xs, "xs")
    if len(source) == 0:
        raise ValueError("xs must contain at least one source row")
    source_y = _labels(ys, len(source), "ys")
    target = _features(xt_labeled, "xt_labeled", source.shape[1])
    target_y = _labels(yt_labeled, len(target), "yt_labeled")
    unlabeled = _features(xu, "xu", source.shape[1])
    scaler = StandardScaler().fit(source)
    source_z = scaler.transform(source)
    rng = np.random.default_rng(seed)
    raw = np.ones(len(source), dtype=np.float64)
    diagnostic = {
        "strategy": strategy, "n_source": len(source),
        "n_target_labels": len(target), "n_unlabeled": len(unlabeled),
        "conflict_fraction": None, "conflict_assessed": False,
        "conflict_probability_cutoff": .1,
        "conflict_interpretation": "target-budget classifier heuristic, not proof of domain conflict",
        "scaler_fit_domain": "source", "status": "ok",
    }
    if strategy == "density":
        covariates = unlabeled if len(unlabeled) else target
        diagnostic["density_target_pool"] = "unlabeled" if len(unlabeled) else "labeled_budget"
        if len(covariates):
            si = rng.choice(len(source), min(len(source), 20000), replace=False)
            ti = rng.choice(len(covariates), min(len(covariates), 20000), replace=False)
            domain_x = np.concatenate([source_z[si], scaler.transform(covariates[ti])])
            domain_y = np.concatenate([np.zeros(len(si)), np.ones(len(ti))])
            domain = LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)
            domain.fit(domain_x, domain_y)
            probability = np.clip(domain.predict_proba(source_z)[:, 1], 1e-6, 1 - 1e-6)
            # Balanced fitting gives equal priors, so no sample-count correction.
            raw = np.clip(probability / (1 - probability), min_weight, max_weight)
            diagnostic.update(domain_fit_source_rows=len(si), domain_fit_target_rows=len(ti),
                              mean_source_target_probability=float(probability.mean()))
        else:
            diagnostic["status"] = "uniform_no_target_covariates"

    if len(np.unique(target_y)) == 2:
        target_model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)
        target_model.fit(scaler.transform(target), target_y)
        attack_probability = target_model.predict_proba(source_z)[:, 1]
        correct_probability = np.where(source_y == 1, attack_probability, 1 - attack_probability)
        conflict = correct_probability < .1
        if strategy == "uncertainty":
            raw = .1 + 4 * attack_probability * (1 - attack_probability)
        raw *= np.where(conflict, np.maximum(correct_probability / .1, .1), 1.0)
        diagnostic.update(conflict_fraction=float(conflict.mean()), conflict_assessed=True,
                          conflicting_source_rows=int(conflict.sum()),
                          mean_target_agreement_probability=float(correct_probability.mean()))
    elif strategy == "uncertainty":
        diagnostic["status"] = "uniform_insufficient_target_classes"

    weights = _bounded_normalize(raw, min_weight, max_weight)
    diagnostic.update(weight_min=float(weights.min()), weight_max=float(weights.max()),
                      weight_mean=float(weights.mean()), weight_sum=float(weights.sum()),
                      effective_source_rows=float(weights.sum() ** 2 / np.square(weights).sum()))
    return weights, diagnostic


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if probabilities.shape != (len(x), len(classes)) or not np.isin(classes, [0, 1]).all():
        raise ValueError("model must return predict_proba columns for binary classes 0/1")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("model returned invalid probabilities")
    positive = np.flatnonzero(classes == 1)
    return probabilities[:, positive[0]] if len(positive) else np.zeros(len(x))


def self_train(model_factory: Callable, x_labeled, y_labeled, sample_weight, x_unlabeled, *,
               iterations: int = 3, confidence: float = .95, pseudo_weight: float = .2,
               max_per_class: int = 1000, seed: int = 42):
    """Refit from original supervision plus unique, cumulative pseudo labels.

    ``max_per_class`` limits new selections per round. Pseudo-label weights are
    at most ``pseudo_weight`` times the mean original weight per row, and their
    cumulative mass is at most ``pseudo_weight * sum(sample_weight)``. A new
    model is created for every refit. The factory must support
    ``fit(X, y, sample_weight=...)`` and binary ``predict_proba``.
    """
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if type(max_per_class) is not int or max_per_class < 1:
        raise ValueError("max_per_class must be a positive integer")
    if not np.isfinite(confidence) or not .5 < confidence <= 1:
        raise ValueError("confidence must lie in (0.5, 1]")
    if not np.isfinite(pseudo_weight) or not 0 < pseudo_weight <= 1:
        raise ValueError("pseudo_weight must lie in (0, 1]")
    x = _features(x_labeled, "x_labeled")
    if not len(x):
        raise ValueError("x_labeled must contain supervised rows")
    y = _labels(y_labeled, len(x), "y_labeled")
    pool = _features(x_unlabeled, "x_unlabeled", x.shape[1])
    weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if (weights.shape != (len(x),) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("sample_weight must be finite, nonnegative, and have positive total mass")
    supervised_mass = float(weights.sum())
    if not np.isfinite(supervised_mass):
        raise ValueError("sample_weight total mass must be finite")
    model = model_factory()
    model.fit(x, y, sample_weight=weights)
    history = []
    selected = np.zeros(len(pool), dtype=bool)
    pseudo_indices = np.empty(0, dtype=np.int64)
    pseudo_labels = np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for iteration in range(1, iterations + 1):
        remaining = np.flatnonzero(~selected)
        if not len(remaining):
            break
        # Predict in bounded batches, including when a large pool remains.
        scores = np.concatenate([_positive_probability(model, pool[batch])
                                 for batch in np.array_split(remaining, max(1, (len(remaining) + 65535) // 65536))])
        candidates = []
        candidate_labels = []
        candidate_confidence = []
        for label in (0, 1):
            certainty = scores if label else 1 - scores
            eligible = np.flatnonzero(certainty >= confidence)
            # Descending confidence with reproducible random tie breaking.
            order = np.lexsort((rng.random(len(eligible)), -certainty[eligible]))
            local = eligible[order[:max_per_class]]
            candidates.extend(remaining[local].tolist())
            candidate_labels.extend([label] * len(local))
            candidate_confidence.extend(certainty[local].tolist())
        new_indices = np.asarray(candidates, dtype=np.int64)
        new_labels = np.asarray(candidate_labels, dtype=np.int64)
        selected[new_indices] = True
        pseudo_indices = np.concatenate([pseudo_indices, new_indices])
        pseudo_labels = np.concatenate([pseudo_labels, new_labels])
        count = len(pseudo_indices)
        per_pseudo_weight = min(pseudo_weight * float(weights.mean()),
                                pseudo_weight * supervised_mass / count) if count else 0.0
        entry = {
            "iteration": iteration, "selected_count": len(new_indices),
            "selected_indices": new_indices.tolist(),
            "selected_class_counts": {str(c): int((new_labels == c).sum()) for c in (0, 1)},
            "total_pseudo_labels": count, "remaining_unlabeled": int((~selected).sum()),
            "supervised_weight_total": supervised_mass,
            "pseudo_weight_total": float(per_pseudo_weight * count),
            "pseudo_weight_cap": float(pseudo_weight * supervised_mass),
            "mean_selected_confidence": float(np.mean(candidate_confidence)) if len(new_indices) else None,
        }
        if not len(new_indices):
            entry["stop_reason"] = "no_confident_predictions"
            history.append(entry)
            break
        model = model_factory()
        model.fit(np.concatenate([x, pool[pseudo_indices]]), np.concatenate([y, pseudo_labels]),
                  sample_weight=np.concatenate([weights, np.full(count, per_pseudo_weight)]))
        if selected.all():
            entry["stop_reason"] = "pool_exhausted"
        history.append(entry)
    return model, history


def entropy_temperature(scores, *, min_temperature: float = .5, max_temperature: float = 2.,
                        steps: int = 40, learning_rate: float = .05,
                        regularization: float = .01):
    """Sharpen binary scores using label-free, regularized entropy minimization.

    This optimizes only one log-temperature, not XGBoost tree parameters. It
    can make incorrect predictions more confident and does not establish or
    repair calibration. Positive temperature preserves ranking and the 0.5
    decision boundary. Evaluation-pool use must be reported as transductive.
    """
    if (not np.isfinite(min_temperature) or not np.isfinite(max_temperature)
            or not 0 < min_temperature <= max_temperature):
        raise ValueError("temperature bounds must be finite and satisfy 0 < minimum <= maximum")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    probability = np.asarray(scores, dtype=np.float64)
    if (probability.ndim != 1 or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("scores must be a vector of finite binary probabilities")
    log_temperature = float(np.clip(0., np.log(min_temperature), np.log(max_temperature)))
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    logits = np.log(clipped) - np.log1p(-clipped)

    def at_temperature(log_t):
        z = np.clip(logits / np.exp(log_t), -700, 700)
        return z, np.clip(1 / (1 + np.exp(-z)), 1e-15, 1 - 1e-15)

    def entropy(p):
        return float(np.mean(-p * np.log(p) - (1 - p) * np.log1p(-p))) if len(p) else 0.

    best_log_t = log_temperature
    best_objective = entropy(at_temperature(log_temperature)[1]) + regularization * log_temperature ** 2
    for _ in range(steps if len(probability) else 0):
        z, current = at_temperature(log_temperature)
        gradient = float(np.mean(z ** 2 * current * (1 - current))) + 2 * regularization * log_temperature
        log_temperature = float(np.clip(log_temperature - learning_rate * gradient,
                                        np.log(min_temperature), np.log(max_temperature)))
        objective = entropy(at_temperature(log_temperature)[1]) + regularization * log_temperature ** 2
        if objective < best_objective:
            best_log_t, best_objective = log_temperature, objective
    adapted = at_temperature(best_log_t)[1]
    return adapted, {
        "method": "label_free_entropy_temperature", "temperature": float(np.exp(best_log_t)),
        "n_unlabeled": len(probability), "entropy_before": entropy(clipped),
        "entropy_after": entropy(adapted), "regularized_objective": float(best_objective),
        "labels_used": 0, "preserves_ranking": True, "preserves_threshold_0_5": True,
        "limitation": "Entropy reduction may amplify mistakes; it is not calibration.",
    }
