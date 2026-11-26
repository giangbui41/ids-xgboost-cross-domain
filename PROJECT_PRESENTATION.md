# Cross-Domain Intrusion Detection with XGBoost
## Project Summary for Professor Presentation

---

## 1. Problem Statement

**Goal**: Build an Intrusion Detection System (IDS) that transfers knowledge from **traditional network traffic datasets** (CIC-IDS 2017, CIC-IDS 2018) to **IoT network traffic** (CICIoT2023).

**Core Challenge**: 
- Source domains (2017, 2018): Traditional IT network traffic with 77+ features
- Target domain (IoT 2023): IoT network traffic with different feature space
- **Only 7 features overlap** across all three domains

**Why this matters**: IoT devices are proliferating but labeled IoT attack data is scarce. Cross-domain transfer learning enables leveraging abundant labeled IT network data for IoT intrusion detection.

---

## 2. The Seven Shared Features (Canonical Feature Set)

| Feature | Description | Domain Mapping |
|---------|-------------|----------------|
| `flow_packets_per_second` | Total packets per second in flow | Flow Bytes/s / Flow Duration |
| `forward_packets_per_second�` | Forward direction packets per second | Fwd Packets/s |
| `backward_packets_per_second` | Backward direction packets per second | Bwd Packets/s |
| `packet_length_min` | Minimum packet length in flow | Min Packet Length |
| `packet_length_mean` | Mean packet length in flow | Mean Packet Length |
| `packet_length_max` | Maximum packet length in flow | Max Packet Length |
| `packet_length_std` | Std deviation of packet lengths | Std Packet Length |

**Critical Design Decision**: Feature mapping is **explicit and auditable** — no heuristic/string matching. Each domain has explicit column names in `config/feature_mapping.json`.

---

## 3. Experimental Protocol (The Rigorous Core)

### Data Separation: DEV vs SUPPLEMENTARY_TEST
```
IoT Data (CICIoT2023)
├── DEV (Development) ──────────────────► Used for: training, threshold selection, model comparison, hyperparameter tuning
└── SUPPLEMENTARY_TEST (Final Holdout) ─► Used ONCE: frozen model evaluation only
```

### Protocol Rules (Strict Enforcement)
| Rule | Enforcement |
|------|-------------|
| **No SUPPLEMENTARY_TEST during development** | CLI `evaluate-final` command requires a frozen model artifact |
| **One-shot final evaluation** | `freeze` command locks model; `evaluate-final` runs exactly once |
| **Threshold fixed on DEV validation** | Cost-optimal threshold selected on source validation, never retuned |
| **Train-median imputation only** | Test/DEV/SUPPLEMENTARY imputed using **only training medians** |
| **SHA256 provenance tracking** | Every artifact records: data hash, mapping hash, config hash, code hash |

### Cost-Sensitive Learning
- **False Positive cost = 100** (alert fatigue, analyst time)
- **False Negative cost = 1** (missed attack)
- **Threshold selection**: Minimize expected cost on validation set
- **Abstention mechanism**: Reject uncertain predictions (target 90% acceptance rate)

---

## 4. Experimental Stages (E0–E8)

| Stage | Name | Purpose | Key Innovation |
|-------|------|---------|----------------|
| **E0** | Smoke Test | Verify pipeline runs end-to-end | Sanity check |
| **E1** | In-Domain Baselines | Train/test on same domain (2017, 2018, IoT) | Native features (77+) vs mapped (7) |
| **E2** | Related Domain | 2017↔2018 cross-domain | Explicit feature mapping + normalization |
| **E3** | Heterogeneous | 2017→IoT, 2018→IoT | 7 shared features only |
| **E4** | Multi-Source | 2017+2018 → IoT | Pooled, weighted, adaptive source combination |
| **E5** | Sensitivity | XGBoost tree depth sweep on E3 | Depth 3, 6, 9 comparison |
| **E6** | Drift Robustness | Label-free measurement drift | Perturb features ±5-20%, measure degradation |
| **E7** | Budgeted Adaptation | Few-shot: K labeled IoT samples per class | K=0,1,2,5,10,20,50,100 |
| **E8** | Improvements | Feature engineering + calibration + TTA + Hybrid DANN+XGBoost | Best-of-all-worlds comparison |

---

## 5. Adaptation Methods (E4, E7, E8)

### Core Methods
| Method | Description | When to Use |
|--------|-------------|-------------|
| `source_only` | Train only on source (2017/2018) | Baseline, no target labels |
| `target_only` | Train only on K labeled target samples | When target labels abundant |
| `pooled` | Concatenate source + target, equal weight | Simple combination |
| `weighted` | α·target + (1-α)·source (α∈(0,1)) | Controlled target influence |

### Advanced Methods (E7/E8)
| Method | Description |
|--------|-------------|
| `weighted_adaptive` | Source weight ∝ 1/density in target feature space (KDE) |
| `self_training` | Pseudo-label confident target predictions, retrain |
| `hybrid_dann_xgb` | Domain-Adversarial Neural Network → frozen encoder → XGBoost |

---

## 6. Key Technical Innovations

### 1. Strict Provenance & Reproducibility
```python
# Every artifact includes:
{
  "data_sha256": "...",
  "mapping_sha256": "...", 
  "config_sha256": "...",
  "code_sha256": "...",
  "seed": 42,
  "protocol": "strict_dev_supplementary_separation"
}
```

### 2. Cost-Sensitive Threshold Selection
```python
def select_threshold(y_val, scores_val):
    # FP_cost=100, FN_cost=1
    costs = fp_cost * FPR + fn_cost * FNR * prevalence
    return threshold_minimizing_cost
```

### 3. Abstention/Rejection Rule
```python
# Predict 1 if p > threshold + margin
# Predict 0 if p < threshold - margin  
# ABSTAIN otherwise (target 90% acceptance)
```

### 4. Hybrid DANN+XGBoost (Optional, requires PyTorch)
- **DANN encoder**: Learns domain-invariant representations
- **Gradient reversal**: Encoder maximizes domain confusion
- **Frozen encoder → XGBoost**: Best of both worlds
- **CPU minibatches**, fixed epochs, no early stopping on holdout

### 5. Feature Engineering (E8)
- Ratios: `fwd/bwd packets`, `fwd/bwd bytes`, `packet_rate/byte_rate`
- Interactions: `length_std × packet_rate`, `length_range × packet_rate`
- Log transforms for heavy-tailed features

### 6. Test-Time Adaptation (TTA)
- **Entropy minimization**: Adapt batch norm to target distribution
- **Self-training**: Pseudo-label confident predictions, retrain

---

## 7. How to Run Experiments

### Prerequisites
```bash
# Install
pip install -e .
# Optional hybrid (DANN)
pip install -e ".[hybrid]"
```

### Data Preparation
```bash
# Prepare source domains (2017, 2018)
python -m ids_xgb.cli prepare-source --domain 2017 --raw-dir /data --output data/cicids2017_balanced_77feats.csv

# Prepare IoT with DEV/SUPPLEMENTARY split
python -m ids_xgb.cli prepare-iot --raw-dir /data --dev-output data/ciciot2023_dev.csv --supplementary-output data/ciciot2023_supplementary_test.csv
```

### Core Pipeline (E0-E6)
```bash
# E1-E4 Matrix
python -m ids_xgb.cli matrix --scenario all --data-dir data --out results/matrix

# E5 Sensitivity (depth sweep)
python -m ids_xgb.cli sensitivity --data-dir data --out results/sensitivity

# E6 Drift
python -m ids_xgb.cli drift --run results/matrix/E3_18toIoT_xgboost_s42 --iot-dev data/ciciot2023_dev.csv --out results/drift
```

### Budgeted Adaptation (E7)
```bash
python -m ids_xgb.cli fewshot --source 2018 --k-list 0 1 2 5 10 20 50 100 --methods source_only weighted self_training --out results/fewshot
```

### Improvements (E8)
```bash
python -m ids_xgb.cli improvements --source 2018 --config config/E8_improved.json --out results/E8
```

### Final Evaluation (ONE SHOT ONLY)
```bash
# 1. Freeze selected model (e.g., from E8)
python -m ids_xgb.cli freeze --run results/E8/weighted_fe_calibrated --out models/final

# 2. ONE evaluation on untouched SUPPLEMENTARY_TEST
python -m ids_xgb.cli evaluate-final --frozen models/final --iot-supplementary data/ciciot2023_supplementary_test.csv --out results/final
```

---

## 8. Expected Results Summary

| Scenario | Expected Macro F1 | Key Insight |
|----------|-------------------|-------------|
| E1 In-Domain (2017, 2018) | >0.95 | Strong baseline on native features |
| E1 In-Domain (IoT) | >0.90 | IoT is learnable with own data |
| E2 Related (17↔18) | >0.85 | High transfer between similar domains |
| E3 Heterogeneous (18→IoT) | ~0.60-0.70 | 7 features insufficient alone |
| E4 Multi-Source + Weighted | ~0.70-0.75 | Combining sources helps |
| E7 Few-Shot (K=10-20) | ~0.75-0.80 | Small target budget helps significantly |
| E8 Hybrid/Improved | ~0.80-0.85 | Best achievable with all techniques |

---

## 9. Presentation Talking Points

### For Your Professor:
1. **Problem is real**: IoT security needs cross-domain transfer; labeled IoT data scarce
2. **Methodology is rigorous**: DEV/SUPPLEMENTARY separation prevents overfitting to test set
3. **Feature mapping is honest**: Explicit 7-feature overlap, no hallucinated equivalence
4. **Cost-sensitive**: Real-world FP/FN costs drive threshold selection
5. **Reproducible**: SHA256 hashes, frozen models, one-shot final evaluation
6. **Comprehensive**: 9 experimental stages covering baselines → adaptation → improvements

### What Makes This Project Strong:
- ✅ **No data leakage** — strict protocol enforced by CLI
- ✅ **Realistic costs** — FP=100, FN=1 reflects operational reality
- ✅ **Ablation study** — E1-E8 systematically isolate each factor
- ✅ **Modern techniques** — DANN, calibration, TTA, feature engineering
- ✅ **Production-ready** — abstention mechanism, provenance tracking

---

## 10. Repository Structure

```
ids-xgboost-cross-domain/
├── config/
│   ├── feature_mapping.json      # 7-feature explicit mapping
│   ├── E0_smoke.json ... E8_improved.json
├── src/ids_xgb/
│   ├── cli.py                    # Main entry point
│   ├── data.py                   # Loading, imputation, SHA256
│   ├── experiment.py             # Training, evaluation, metrics
│   ├── pipeline.py               # E0-E8 orchestration
│   ├── matrix.py                 # E1-E4 scenarios
│   ├── fewshot.py                # E7 budgeted adaptation
│   ├── improvements.py           # E8 runner
│   ├── adaptation.py             # Adaptive weighting, self-training
│   ├── calibration.py            # Platt/Isotonic
│   ├── hybrid.py                 # DANN+XGBoost
│   ├── features.py               # Feature engineering
│   ├── explanations.py           # TreeSHAP
│   └── robustness.py             # Drift measurement
├── pyproject.toml
└── README.md
```

---

## 11. Key Files to Show Professor

1. **`config/feature_mapping.json`** — The honest 7-feature overlap
2. **`src/ids_xgb/data.py`** — Train-median imputation, SHA256, protocol
3. **`src/ids_xgb/experiment.py`** — Cost-sensitive threshold, abstention
4. **`src/ids_xgb/pipeline.py`** — E0-E8 orchestration with registry
5. **`src/ids_xgb/hybrid.py`** — DANN+XGBoost architecture
6. **`src/ids_xgb/cli.py`** — Commands enforcing protocol

---

*Prepared for presentation on 2026-11-27*
*Project: ids-xgboost-cross-domain*