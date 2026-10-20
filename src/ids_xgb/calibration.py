"""Probability calibration with explicit held-out labels and auditable metrics."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def calibration_metrics(y, scores, *, n_bins=10, tolerance=.05):
    y, p = np.asarray(y), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or p.shape != y.shape or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()):
        raise ValueError("Calibration requires binary labels and finite probabilities in [0,1]")
    if type(n_bins) is not int or n_bins < 2 or not 0 <= tolerance <= 1:
        raise ValueError("Invalid calibration bins/tolerance")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    reliability = []
    ece = 0.
    for index in range(n_bins):
        selected = bins == index
        if selected.any():
            predicted, observed = float(p[selected].mean()), float(y[selected].mean())
            ece += selected.mean() * abs(predicted - observed)
            reliability.append({"bin": index, "count": int(selected.sum()),
                                "mean_probability": predicted, "attack_fraction": observed})
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {"ece": float(ece), "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "is_well_calibrated": bool(ece <= tolerance), "tolerance": tolerance,
            "n_bins": n_bins, "n": len(y), "reliability": reliability}


def check_calibration(model, X_val, y_val, *, n_bins=10, tolerance=.05):
    """Descriptive ECE/Brier/log loss; the ECE cutoff is a heuristic, not a test."""
    return calibration_metrics(y_val, model.predict_proba(X_val)[:, 1],
                               n_bins=n_bins, tolerance=tolerance)


class CalibratedModel:
    """A fitted base model plus a mapping learned on separate calibration rows."""

    def __init__(self, model, calibrator, method):
        self.model = model
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    @staticmethod
    def logits(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
        return (np.log(p) - np.log1p(-p)).reshape(-1, 1)

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            p = self.calibrator.predict_proba(self.logits(p))[:, 1]
        else:
            p = self.calibrator.predict(p)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def calibrate_prefit(model, X_calibration, y_calibration, *, method="sigmoid"):
    """Caller must supply rows excluded from base-model fitting and evaluation."""
    y = np.asarray(y_calibration)
    p = model.predict_proba(X_calibration)[:, 1]
    calibration_metrics(y, p)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration fitting requires both classes")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Calibration requires classes [0, 1]")
    if method == "sigmoid":
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibrator.fit(CalibratedModel.logits(p), y)
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    else:
        raise ValueError("Calibration method must be sigmoid or isotonic")
    return CalibratedModel(model, calibrator, method)


def calibrate_model(model, X_train, y_train, cv=5, *, method="sigmoid", sample_weight=None):
    """Guide-compatible cross-validation API; refits clones on training data only.

    X_train must already be finite. For learned preprocessing use an estimator
    pipeline so each fold fits its own transforms. The experiment runner uses
    calibrate_prefit with a separately reserved, budgeted calibration split.
    """
    if method not in {"sigmoid", "isotonic"} or type(cv) is not int or cv < 2:
        raise ValueError("Use sigmoid/isotonic and integer cv >= 2")
    y = np.asarray(y_train)
    if set(np.unique(y)) != {0, 1} or min(np.bincount(y.astype(int))) < cv:
        raise ValueError("Each class must have at least cv training rows")
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X_train, y, sample_weight=sample_weight)
    return calibrated
