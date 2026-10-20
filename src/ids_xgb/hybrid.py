"""Optional DANN representation followed by a frozen XGBoost classifier.

Only source labels and explicitly budgeted target labels enter the task loss.
Target adaptation features (with no labels) enter the domain loss. Callers must
keep all validation/final holdout rows outside ``fit``. Training uses fixed
epochs, without holdout-based early stopping. This is an experimental baseline,
not a guarantee of improvement over raw-feature XGBoost.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "HybridDANNXGB requires optional PyTorch. Install the project's "
            "hybrid extra with `python -m pip install -e '.[hybrid]'`, or "
            "install a suitable PyTorch build from https://pytorch.org/get-started/locally/."
        ) from exc
    return torch


def _reverse_gradient(value, coefficient: float):
    """Identity in the forward pass, -coefficient times the backward gradient.

    Detaching the identity term avoids a custom, locally defined autograd class
    and makes the encoder a plain serializable torch.nn.Sequential module.
    """
    detached = value.detach()
    return detached + coefficient * (detached - value)


def _weighted_task_loss(torch, logits, labels, weights, batch_divisor):
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    # A constant divisor also covers the last, shorter batch: otherwise a rare
    # high-weight target row there would be amplified by batch_size / len(batch).
    return (losses * weights).sum() / batch_divisor


def _matrix(values, name: str, columns: int | None = None, *, allow_empty=False):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} features")
    if not allow_empty and len(array) == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values; impute before fitting")
    return np.ascontiguousarray(array)


def _labels(values, count: int, name: str):
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != count or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain one binary label (0 or 1) per row")
    return array.astype(np.int64)


class HybridDANNXGB:
    """Source-scaled DANN + XGBoost, with CPU minibatches and optional torch.

    ``sample_weight`` follows source rows then labeled-target rows and affects
    both the supervised neural loss and final XGBoost fit. Domain minibatches
    are balanced between source and target and use no task labels. At most
    ``max_unlabeled`` adaptation rows are sampled once, with the given seed.
    The labeled pool is visited once per epoch; all neural passes are batched.
    """

    def __init__(self, seed=42, jobs=4, xgb_params=None, hidden_dim=32,
                 embedding_dim=16, epochs=10, batch_size=256,
                 learning_rate=.001, lambda_domain=.1, max_unlabeled=20000):
        self.seed = seed
        self.jobs = jobs
        self.xgb_params = dict(xgb_params or {})
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_domain = lambda_domain
        self.max_unlabeled = max_unlabeled
        self._validate_config()

    def _validate_config(self):
        for name in ("hidden_dim", "embedding_dim", "epochs", "batch_size", "max_unlabeled"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(self.jobs, bool) or not isinstance(self.jobs, (int, np.integer)) or self.jobs == 0:
            raise ValueError("jobs must be a nonzero integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.lambda_domain) or self.lambda_domain < 0:
            raise ValueError("lambda_domain must be finite and nonnegative")
        if any(self.xgb_params.get(key) is not None for key in ("early_stopping_rounds", "callbacks")):
            raise ValueError("Hybrid fitting uses fixed training rounds, without early stopping or callbacks")
        if self.xgb_params.get("objective", "binary:logistic") != "binary:logistic":
            raise ValueError("HybridDANNXGB requires objective='binary:logistic'")

    def _build_encoder(self, torch):
        return torch.nn.Sequential(
            torch.nn.Linear(self.n_features_in_, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.embedding_dim),
            torch.nn.ReLU(),
        )

    def fit(self, xs, ys, xt_labeled, yt_labeled, x_unlabeled, sample_weight=None):
        """Fit using source, budgeted target, and unlabeled adaptation rows.

        Empty target arrays must retain shape ``(0, n_features)``. No holdout
        or unlabeled-target label argument is accepted by this interface.
        """
        self._validate_config()
        xs = _matrix(xs, "xs")
        xt = _matrix(xt_labeled, "xt_labeled", xs.shape[1], allow_empty=True)
        xu = _matrix(x_unlabeled, "x_unlabeled", xs.shape[1], allow_empty=True)
        ys = _labels(ys, len(xs), "ys")
        yt = _labels(yt_labeled, len(xt), "yt_labeled")
        labels = np.concatenate([ys, yt])
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Supervised source/target rows must include both binary classes")
        if len(xt) + len(xu) == 0:
            raise ValueError("DANN requires labeled or unlabeled target adaptation rows")
        weights = np.ones(len(labels), dtype=np.float32)
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float32)
            if (weights.shape != labels.shape or not np.isfinite(weights).all()
                    or (weights < 0).any() or weights.sum(dtype=np.float64) <= 0):
                raise ValueError("sample_weight must be finite, nonnegative, and match supervised rows")
        if any(weights[labels == label].sum(dtype=np.float64) <= 0 for label in (0, 1)):
            raise ValueError("Both supervised classes need positive total sample weight")

        torch = _require_torch()
        from xgboost import XGBClassifier

        self.n_features_in_ = xs.shape[1]
        self.embedding_feature_names_ = [f"dann_embedding_{i:02d}" for i in range(self.embedding_dim)]
        rng = np.random.default_rng(self.seed)
        if len(xu) > self.max_unlabeled:
            xu = xu[rng.choice(len(xu), self.max_unlabeled, replace=False)]
        self.n_unlabeled_used_ = len(xu)
        self.scaler_ = StandardScaler().fit(xs)
        supervised = np.concatenate([xs, xt])
        domain_target = np.concatenate([xt, xu])
        x_task = torch.from_numpy(self.scaler_.transform(supervised).astype(np.float32))
        x_target = torch.from_numpy(self.scaler_.transform(domain_target).astype(np.float32))
        if not torch.isfinite(x_task).all() or not torch.isfinite(x_target).all():
            raise ValueError("Source standardization produced nonfinite values; inspect feature scales")
        y_task = torch.from_numpy(labels.astype(np.float32))
        # A fixed normalization preserves relative weights across minibatches.
        w_task = torch.from_numpy((weights / weights.mean(dtype=np.float64)).astype(np.float32))
        self.history_ = []
        old_threads = torch.get_num_threads()
        try:
            if self.jobs > 0:
                torch.set_num_threads(self.jobs)
            # Preserve the process's CPU RNG state. All modules and data stay on CPU.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self.encoder_ = self._build_encoder(torch)
                task_head = torch.nn.Linear(self.embedding_dim, 1)
                domain_head = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_dim, self.hidden_dim),
                    torch.nn.ReLU(), torch.nn.Linear(self.hidden_dim, 1),
                )
                parameters = (list(self.encoder_.parameters()) + list(task_head.parameters())
                              + list(domain_head.parameters()))
                optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
                half_batch = max(1, self.batch_size // 2)
                task_batch_size = min(self.batch_size, len(labels))
                for epoch in range(self.epochs):
                    task_sum = domain_sum = accuracy_sum = 0.0
                    permutation = rng.permutation(len(labels))
                    for start in range(0, len(labels), self.batch_size):
                        index = torch.from_numpy(permutation[start:start + self.batch_size])
                        optimizer.zero_grad(set_to_none=True)
                        task_logits = task_head(self.encoder_(x_task[index])).squeeze(1)
                        task_loss = _weighted_task_loss(
                            torch, task_logits, y_task[index], w_task[index], task_batch_size)
                        source_index = torch.from_numpy(rng.integers(len(xs), size=half_batch))
                        target_index = torch.from_numpy(rng.integers(len(x_target), size=half_batch))
                        domain_x = torch.cat([x_task[source_index], x_target[target_index]])
                        domain_y = torch.cat([torch.zeros(half_batch), torch.ones(half_batch)])
                        embedding = _reverse_gradient(self.encoder_(domain_x), self.lambda_domain)
                        domain_logits = domain_head(embedding).squeeze(1)
                        domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            domain_logits, domain_y)
                        # The domain head minimizes BCE; gradient reversal makes
                        # the encoder maximize lambda_domain * domain BCE.
                        loss = task_loss + domain_loss * len(index) / task_batch_size
                        if not torch.isfinite(loss):
                            raise ValueError("DANN loss became nonfinite; inspect data or lower learning_rate")
                        loss.backward()
                        optimizer.step()
                        task_sum += float(task_loss.detach()) * task_batch_size
                        domain_sum += float(domain_loss.detach()) * len(index)
                        accuracy_sum += float(((domain_logits.detach() >= 0) == domain_y.bool())
                                              .float().mean()) * len(index)
                    self.history_.append({"epoch": epoch + 1,
                                          "task_loss": task_sum / len(labels),
                                          "domain_loss": domain_sum / len(labels),
                                          "domain_accuracy": accuracy_sum / len(labels)})
                self.encoder_.eval()
                for parameter in self.encoder_.parameters():
                    parameter.requires_grad_(False)
        finally:
            if self.jobs > 0:
                torch.set_num_threads(old_threads)
        params = dict(n_estimators=500, max_depth=6, learning_rate=.05,
                      subsample=.8, colsample_bytree=.8, min_child_weight=1,
                      reg_lambda=1.0, eval_metric="logloss", tree_method="hist")
        params.update(self.xgb_params)
        params.update(objective="binary:logistic", n_jobs=self.jobs, random_state=self.seed)
        self.classifier_ = XGBClassifier(**params)
        self.classifier_.fit(self.transform(supervised), labels, sample_weight=weights)
        self.classes_ = self.classifier_.classes_
        return self

    def transform(self, x):
        """Apply the source-fitted scaler and frozen encoder in bounded batches."""
        if not hasattr(self, "encoder_"):
            raise ValueError("HybridDANNXGB is not fitted")
        x = _matrix(x, "x", self.n_features_in_, allow_empty=True)
        torch = _require_torch()
        output = np.empty((len(x), self.embedding_dim), dtype=np.float32)
        self.encoder_.eval()
        old_threads = torch.get_num_threads()
        try:
            if self.jobs > 0:
                torch.set_num_threads(self.jobs)
            with torch.no_grad():
                for start in range(0, len(x), self.batch_size):
                    batch = self.scaler_.transform(x[start:start + self.batch_size]).astype(np.float32)
                    output[start:start + len(batch)] = self.encoder_(torch.from_numpy(batch)).numpy()
        finally:
            if self.jobs > 0:
                torch.set_num_threads(old_threads)
        if not np.isfinite(output).all():
            raise ValueError("DANN produced nonfinite embeddings; inspect feature scales")
        return output

    def predict_proba(self, x):
        if not hasattr(self, "classifier_"):
            raise ValueError("HybridDANNXGB is not fitted")
        embedding = self.transform(x)
        if len(embedding) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return self.classifier_.predict_proba(embedding)

    def save(self, out_dir):
        """Save a tensor state dict, JSON scaler, XGBoost and configuration.

        No pickled model object or locally defined class is serialized.
        """
        if not hasattr(self, "classifier_"):
            raise ValueError("HybridDANNXGB is not fitted")
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch = _require_torch()
        torch.save(self.encoder_.state_dict(), out_dir / "dann_encoder.pt")
        self.classifier_.save_model(str(out_dir / "xgboost.json"))
        scaler = {"mean": self.scaler_.mean_.tolist(), "scale": self.scaler_.scale_.tolist(),
                  "var": self.scaler_.var_.tolist(), "n_samples_seen": int(self.scaler_.n_samples_seen_)}
        config = {"format_version": 1, "seed": int(self.seed), "jobs": int(self.jobs),
                  "xgb_params": self.xgb_params, "hidden_dim": int(self.hidden_dim),
                  "embedding_dim": int(self.embedding_dim), "epochs": int(self.epochs),
                  "batch_size": int(self.batch_size), "learning_rate": float(self.learning_rate),
                  "lambda_domain": float(self.lambda_domain), "max_unlabeled": int(self.max_unlabeled),
                  "n_features_in": int(self.n_features_in_), "n_unlabeled_used": self.n_unlabeled_used_,
                  "embedding_feature_names": self.embedding_feature_names_, "history": self.history_,
                  "protocol": "source-scaler; fixed-epochs; budgeted-target-labels; unlabeled-adaptation-only"}
        for filename, value in (("scaler.json", scaler), ("hybrid.json", config)):
            (out_dir / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, out_dir):
        """Restore locally produced artifacts without unpickling a model object."""
        out_dir = Path(out_dir)
        config = json.loads((out_dir / "hybrid.json").read_text(encoding="utf-8"))
        if config.get("format_version") != 1:
            raise ValueError("Unsupported hybrid artifact version")
        keys = ("seed", "jobs", "xgb_params", "hidden_dim", "embedding_dim", "epochs",
                "batch_size", "learning_rate", "lambda_domain", "max_unlabeled")
        model = cls(**{key: config[key] for key in keys})
        model.n_features_in_ = config["n_features_in"]
        model.n_unlabeled_used_ = config["n_unlabeled_used"]
        model.embedding_feature_names_ = config["embedding_feature_names"]
        model.history_ = config["history"]
        scaler = json.loads((out_dir / "scaler.json").read_text(encoding="utf-8"))
        model.scaler_ = StandardScaler()
        model.scaler_.mean_ = np.asarray(scaler["mean"], dtype=np.float64)
        model.scaler_.scale_ = np.asarray(scaler["scale"], dtype=np.float64)
        model.scaler_.var_ = np.asarray(scaler["var"], dtype=np.float64)
        model.scaler_.n_samples_seen_ = scaler["n_samples_seen"]
        model.scaler_.n_features_in_ = model.n_features_in_
        torch = _require_torch()
        with torch.random.fork_rng(devices=[]):
            model.encoder_ = model._build_encoder(torch)
        state = torch.load(out_dir / "dann_encoder.pt", map_location="cpu", weights_only=True)
        model.encoder_.load_state_dict(state)
        model.encoder_.eval()
        for parameter in model.encoder_.parameters():
            parameter.requires_grad_(False)
        from xgboost import XGBClassifier
        model.classifier_ = XGBClassifier(n_jobs=model.jobs)
        model.classifier_.load_model(str(out_dir / "xgboost.json"))
        model.classes_ = model.classifier_.classes_
        return model
