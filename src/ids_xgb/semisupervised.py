"""Self-training API for source, budgeted target labels and unlabeled target."""
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from .adaptation import self_train
from .experiment import make_model


def fit_self_training(X_source, y_source, X_target_labeled, y_target_labeled,
                      X_target_unlabeled, model_type="xgboost", iterations=3, *,
                      seed=42, jobs=4, confidence_threshold=.95, pseudo_weight=.2,
                      max_per_class=1000):
    """Return final model and fitted models; inputs must be finite matrices.

    Unlike the illustrative guide loop, each pseudo row is selected only once
    and pseudo-label weights are bounded by the true-label weight mass.
    """
    if model_type not in {"xgboost", "random_forest"}:
        raise ValueError("Self-training supports xgboost or random_forest")
    x = np.concatenate([X_source, X_target_labeled])
    y = np.concatenate([y_source, y_target_labeled])
    models = []

    def factory():
        model = make_model(model_type, seed, jobs)
        models.append(model)
        return model

    model, history = self_train(factory, x, y, compute_sample_weight("balanced", y),
                               X_target_unlabeled, iterations=iterations,
                               confidence=confidence_threshold, pseudo_weight=pseudo_weight,
                               max_per_class=max_per_class, seed=seed)
    model.self_training_history_ = history
    return model, models
