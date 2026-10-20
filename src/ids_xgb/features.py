"""Deterministic shared-feature transforms and explicit causal history features.

The row-wise engineer learns a column schema only, never population statistics.
Temporal features require real timestamps and entity/session metadata; the
prepared seven-column CSVs alone do not contain enough information to use them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CANONICAL_FEATURES = (
    "flow_packets_per_second", "forward_packets_per_second",
    "backward_packets_per_second", "packet_length_min", "packet_length_mean",
    "packet_length_max", "packet_length_std",
)


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise ValueError("Features must be a DataFrame with unique column names")
    if any(not isinstance(name, str) for name in frame.columns):
        raise ValueError("Feature names must be strings")
    if any(name.lower() in {"label", "target", "y"} for name in frame.columns):
        raise ValueError("Label columns must not be passed to feature engineering")
    result = frame.apply(pd.to_numeric, errors="coerce").astype("float64")
    # Values will eventually be converted to float32 for model fitting. Undefined
    # arithmetic remains missing and is imputed by the training-only imputer.
    return result.where(np.isfinite(result) & (result.abs() <= np.finfo(np.float32).max))


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.div(denominator.where(denominator > 0))
    return result.mask((numerator == 0) & (denominator == 0), 0.0)


class FeatureEngineer:
    """Add available ratios/interactions/logs while retaining original columns.

    ``fit`` records the schema and does not estimate means, thresholds, medians,
    or any target statistics. Subsets such as ``length_only`` are supported.
    ``temporal`` is deliberately rejected when enabled: call
    :func:`causal_temporal_features` with explicit metadata before this class.
    """

    def __init__(self, enabled: bool = True, temporal=None):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        if temporal is not None and temporal is not False:
            raise ValueError("Use causal_temporal_features with real timestamps and group metadata")
        self.enabled = enabled
        self.temporal = temporal

    def fit(self, frame: pd.DataFrame):
        numeric = _numeric(frame)
        self.feature_names_in_ = list(numeric.columns)
        self.feature_names_out_ = list(self._expand(numeric.iloc[:0]).columns)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_in_"):
            raise ValueError("FeatureEngineer must be fitted before transform")
        if list(frame.columns) != self.feature_names_in_:
            raise ValueError("Feature columns or their order differ from the fitted schema")
        return self._expand(_numeric(frame))

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def _expand(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return frame.copy()
        result = frame.copy()
        # Rates and lengths are nonnegative quantities; invalid negative inputs
        # are missing only in their derived features, leaving originals auditable.
        clean = frame.where(frame >= 0)

        def add(name, values):
            name = "eng_" + name
            if name in result.columns:
                raise ValueError(f"Engineered feature collides with input column: {name}")
            result[name] = values

        for name in CANONICAL_FEATURES:
            if name in clean:
                add("log1p_" + name, np.log1p(clean[name]))
        flow = "flow_packets_per_second"
        forward, backward = "forward_packets_per_second", "backward_packets_per_second"
        minimum, mean, maximum, std = (
            "packet_length_min", "packet_length_mean", "packet_length_max", "packet_length_std")
        ratio_specs = (
            ("forward_fraction", forward, flow),
            ("backward_fraction", backward, flow),
            ("forward_backward_ratio", forward, backward),
            ("length_coefficient_of_variation", std, mean),
            ("length_min_max_ratio", minimum, maximum),
            ("length_mean_max_ratio", mean, maximum),
        )
        for name, numerator, denominator in ratio_specs:
            if numerator in clean and denominator in clean:
                add(name, _ratio(clean[numerator], clean[denominator]))
        if minimum in clean and maximum in clean:
            span = (clean[maximum] - clean[minimum]).where(clean[maximum] >= clean[minimum])
            add("length_span", span)
            if mean in clean:
                add("length_span_over_mean", _ratio(span, clean[mean]))
        if forward in clean and backward in clean:
            total = clean[forward] + clean[backward]
            add("directional_imbalance", _ratio(clean[forward] - clean[backward], total))
        if flow in clean and mean in clean:
            with np.errstate(over="ignore", invalid="ignore"):
                add("rate_length_interaction", clean[flow] * clean[mean])
        if flow in clean and std in clean:
            with np.errstate(over="ignore", invalid="ignore"):
                add("rate_length_std_interaction", clean[flow] * clean[std])
        return result.where(np.isfinite(result) & (result.abs() <= np.finfo(np.float32).max))


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Stateless row-wise feature engineering, as used in the integration guide."""
    return FeatureEngineer().fit_transform(frame)


def causal_temporal_features(frame: pd.DataFrame, metadata: pd.DataFrame,
                             time_column: str, group_columns: list[str],
                             window: int = 5) -> pd.DataFrame:
    """Append prior-only rolling mean/std, prior count, and gap in seconds.

    History is sorted by real event time within each entity/session group and
    excludes the current event. Numeric times are seconds; strings are parsed
    as UTC timestamps. Equal timestamps within a group are rejected because
    their ordering is ambiguous. The returned rows keep the input index/order.

    Call on a chronological stream with only history available at deployment.
    Do not compute history across randomly interleaved train/holdout rows: such
    splits do not establish the causal availability of holdout history.
    """
    numeric = _numeric(frame)
    if not isinstance(metadata, pd.DataFrame) or not frame.index.equals(metadata.index):
        raise ValueError("Temporal metadata must be a DataFrame aligned exactly with feature rows")
    if metadata.columns.has_duplicates:
        raise ValueError("Temporal metadata columns must be unique")
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    if (not isinstance(group_columns, (list, tuple)) or not group_columns
            or len(set(group_columns)) != len(group_columns) or time_column in group_columns):
        raise ValueError("Provide distinct entity/session group columns separate from event time")
    required = [*group_columns, time_column]
    if any(name not in metadata for name in required):
        raise ValueError("Temporal features require real time and group metadata columns")
    if metadata.loc[:, required].isna().any().any():
        raise ValueError("Temporal time/group metadata cannot contain missing values")
    times = metadata[time_column]
    if pd.api.types.is_numeric_dtype(times):
        seconds = times.to_numpy(dtype=float)
    else:
        try:
            parsed = pd.to_datetime(times, utc=True, errors="raise")
            seconds = np.asarray((parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds())
        except (ValueError, TypeError) as exc:
            raise ValueError("Temporal timestamps must be numeric seconds or parseable timestamps") from exc
    if not np.isfinite(seconds).all():
        raise ValueError("Temporal timestamps must be finite and nonmissing")
    keys = metadata.loc[:, group_columns].reset_index(drop=True).copy()
    time_key = "__ids_event_time__"
    while time_key in keys:
        time_key += "_"
    keys[time_key] = seconds
    if keys.duplicated(subset=[*group_columns, time_key]).any():
        raise ValueError("Equal timestamps within a group have ambiguous causal order")
    values = numeric.to_numpy()
    prior_mean = np.full(values.shape, np.nan)
    prior_std = np.full(values.shape, np.nan)
    gap = np.full(len(frame), np.nan)
    count = np.zeros(len(frame), dtype=int)
    for positions in keys.groupby(list(group_columns), sort=False, observed=True).indices.values():
        ordered = positions[np.argsort(seconds[positions], kind="stable")]
        history = pd.DataFrame(values[ordered]).shift(1).rolling(window, min_periods=1)
        prior_mean[ordered] = history.mean().to_numpy()
        prior_std[ordered] = history.std(ddof=0).to_numpy()
        gap[ordered[1:]] = np.diff(seconds[ordered])
        count[ordered] = np.minimum(np.arange(len(ordered)), window)
    result = numeric.copy()
    additions = {"temporal_prior_count": count, "temporal_gap_seconds": gap}
    for column, name in enumerate(numeric.columns):
        additions[f"temporal_prior_mean_{name}"] = prior_mean[:, column]
        additions[f"temporal_prior_std_{name}"] = prior_std[:, column]
    if set(additions).intersection(result.columns):
        raise ValueError("Temporal output names collide with existing feature columns")
    return pd.concat([result, pd.DataFrame(additions, index=frame.index)], axis=1).replace(
        [np.inf, -np.inf], np.nan)
