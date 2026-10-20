"""Small executable examples; all rows are synthetic and no dataset is opened."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
from ids_xgb.features import FeatureEngineer, causal_temporal_features
from ids_xgb.experiment import metrics
from ids_xgb.features import engineer_features
from ids_xgb.experiment import metrics_with_cost, metrics_cost_sensitive
from ids_xgb.calibration import calibrate_model, check_calibration
from ids_xgb.semisupervised import fit_self_training


def main():
    frame = pd.DataFrame({"flow_packets_per_second": [10., 20., 12.],
                          "forward_packets_per_second": [7., 15., 8.],
                          "backward_packets_per_second": [3., 5., 4.],
                          "packet_length_min": [20., 25., 20.],
                          "packet_length_mean": [50., 60., 55.],
                          "packet_length_max": [80., 100., 90.],
                          "packet_length_std": [10., 20., 15.]})
    expanded = FeatureEngineer().fit_transform(frame)
    print(expanded[["eng_forward_fraction", "eng_rate_length_interaction"]])
    metadata = pd.DataFrame({"device": ["example-device"] * 3, "event_time_seconds": [0., 2., 5.]})
    temporal = causal_temporal_features(frame, metadata, "event_time_seconds", ["device"], window=2)
    print(temporal[["temporal_prior_count", "temporal_gap_seconds", "temporal_prior_mean_flow_packets_per_second"]])
    report = metrics(np.array([0, 1, 1]), np.array([.8, .9, .2]), .5)
    print({key: report[key] for key in ("total_cost", "cost_per_sample", "macro_f1")})


if __name__ == "__main__":
    main()
