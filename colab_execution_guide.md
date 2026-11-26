# Colab Execution Guide for Cross-Domain IDS Project

This guide walks you through running the complete experimental pipeline (E0–E7) on Google Colab with the fixed codebase.

---

## 1. Environment Setup

### 1.1 Open Colab and Configure Runtime
1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook: **File → New notebook**
3. Set GPU runtime: **Runtime → Change runtime type → T4 GPU** (recommended for XGBoost training speed)

### 1.2 Install System Dependencies
```bash
# Run in a Colab cell
!apt-get update -qq && apt-get install -y -qq git python3.12-venv 2>/dev/null | tail -5
```

### 1.3 Clone Repository
```bash
# Replace with your actual repo URL
!git clone https://github.com/YOUR_USERNAME/ids-xgboost-cross-domain.git
%cd ids-xgboost-cross-domain
```

### 1.4 Create Virtual Environment & Install Package
```bash
!python3.12 -m venv .venv
!source .venv/bin/activate && pip install --upgrade pip setuptools wheel
!source .venv/bin/activate && pip install -e ".[dev]"
```

> **Note**: The `.[dev]` extra installs development dependencies (pytest, black, mypy, etc.). For minimal install, use `pip install -e .`

### 1.5 Verify Installation
```bash
!source .venv/bin/activate && python -c "import ids_xgb; print('ids_xgb version:', ids_xgb.__version__)"
!source .venv/bin/activate && python -m ids_xgb --help
```

---

## 2. Data Preparation

### 2.1 Expected Directory Structure
```
data/
├── cicids2017_balanced_77feats.csv      # CIC-IDS2017 (77 features)
├── cleaned_data_sampled.csv              # CIC-IDS2018 (77 features)
├── ciciot2023_dev.csv                    # CICIoT2023 dev set (7 canonical features)
└── ciciot2023_supplementary_test.csv     # CICIoT2023 SUPPLEMENTARY test set (7 canonical features)
```

### 2.2 Upload Data to Colab

**Option A: Google Drive Mount (Recommended for large files)**
```bash
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive to local data directory
!mkdir -p data
!cp /content/drive/MyDrive/ids_data/* data/
!ls -la data/
```

**Option B: Direct Upload (for smaller files)**
```bash
# Use the file upload widget in Colab sidebar, then:
!mkdir -p data
!mv *.csv data/ 2>/dev/null || true
!ls -la data/
```

**Option C: Download from Source (if publicly available)**
```bash
!mkdir -p data
# Add wget/curl commands here if datasets have public URLs
```

### 2.3 Verify Data Files
```bash
!source .venv/bin/activate && python -m ids_xgb audit --data-dir data --mapping config/feature_mapping.json
```
This runs **E0_data_audit** and validates all 4 CSV files have expected columns, labels, and feature overlap.

---

## 3. Configuration Files

The `config/` directory contains JSON configs for each stage. Key files:

| Config File | Stage | Purpose |
|-------------|-------|---------|
| `E0_data_audit.json` | E0 | Data audit (no config needed) |
| `E1_in_domain.json` | E1 | In-domain baselines (2017, 2018, IoT) |
| `E2_related_transfer.json` | E2 | Related-domain transfer (2017↔2018) |
| `E3_zero_shot_iot.json` | E3 | Zero-shot to IoT (2017→IoT, 2018→IoT) |
| `E4_fewshot.json` | E4 | Few-shot adaptation (K=10, 50, 200) |
| `E5_alpha_sensitivity.json` | E5 | Alpha sensitivity for weighted method |
| `E6_measurement_drift.json` | E6 | Measurement drift robustness (ε=0.05, 0.10) |
| `E7_final_evaluation.json` | E7 | **Locked** supplementary evaluation |
| `E8_improvements.json` | E8 | Improved methods (self-training, calibration, hybrid) |

### 3.1 Inspect a Config (Example: E4)
```bash
!cat config/E4_fewshot.json
```

Key parameters to understand:
- `seeds`: Random seeds for repeated runs (e.g., `[42, 123, 456]`)
- `sources`: Source domains (`["2017", "2018"]`)
- `k_per_class`: Labels per class for few-shot (e.g., `[10, 50, 200]`)
- `methods`: Adaptation methods (`["source_only", "weighted", "pooled", "target_only"]`)
- `alpha`: Weight for target domain in weighted method (0.5 default)
- `holdout_fraction`: Fraction of target dev for holdout (0.4 default)
- `model`: Model type (`"xgboost"`, `"random_forest"`, `"logistic_regression"`)
- `group`: Feature group (`"seven"` for 7 canonical features)

---

## 4. Running the Pipeline (E0 → E7)

### 4.1 Stage E0: Data Audit
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E0 \
  --data-dir data --results-dir results --mapping config/feature_mapping.json
```
**Output**: `results/current/E0.csv` with per-feature statistics for all 3 datasets.

---

### 4.2 Stage E1: In-Domain Baselines
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E1 \
  --data-dir data --results-dir results --mapping config/feature_mapping.json
```
**Output**: `results/current/E1.csv` — in-domain performance on 2017, 2018, IoT dev.

---

### 4.3 Stage E2: Related-Domain Transfer
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E2 \
  --data-dir data --results-dir results --mapping config/feature_mapping.json
```
**Output**: `results/current/E2.csv` — 2017→2018 and 2018→2017 transfer.

---

### 4.4 Stage E3: Zero-Shot to IoT
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E3 \
  --data-dir data --results-dir results --mapping config/feature_mapping.json
```
**Output**: `results/current/E3.csv` — 2017→IoT and 2018→IoT zero-shot.

---

### 4.5 Stage E4: Few-Shot Adaptation
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E4 \
  --data-dir data --results-dir results --mapping config/feature_mapping.json
```
**Output**: `results/current/E4.csv` — Few-shot results for K=10, 50, 200 across methods.

> **Time estimate**: ~10-30 minutes depending on seeds and K values.

---

### 4.6 Stage E5: Alpha Sensitivity (Weighted Method)
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E5 \
  --data-dir data --results-dir results --mapping config/feature_mapping.json
```
**Output**: `results/current/E5.csv` — Performance vs. alpha (target weight).

---

### 4.7 Stage E6: Measurement Drift Robustness
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E6 \
  --data-dir data --results-dir results --mapping config/feature_mapping.json
```
**Output**: `results/current/E6.csv` — Robustness to feature perturbation (ε=0.05, 0.10).

---

### 4.8 Stage E7: Supplementary Evaluation (LOCKED)

**⚠️ CRITICAL**: E7 uses a **lock-final** mechanism. You must:
1. Select your best configuration from E4/E5/E6 results
2. Create a locked E7 config with that selection
3. Run `lock-final` **once** (creates immutable lock file)
4. Run E7 (reads SUPPLEMENTARY only after lock verification)

#### Step 1: Pick Best Config from E4/E5/E6
```bash
# View aggregate results
!source .venv/bin/activate && python -c "
import pandas as pd
for stage in ['E4', 'E5', 'E6']:
    df = pd.read_csv(f'results/current/{stage}.csv')
    print(f'\n=== {stage} ===')
    print(df.sort_values('macro_f1_mean', ascending=False).head(10)[['source','method','model','k_per_class','alpha','macro_f1_mean','macro_f1_std']])
"
```

#### Step 2: Create E7 Config with Your Selection
Edit `config/E7_final_evaluation.json` with your chosen params:
```json
{
  "stage": "E7",
  "source": "2018",
  "method": "weighted",
  "k_per_class": 50,
  "alpha": 0.5,
  "model": "xgboost",
  "seed": 42,
  "group": "seven",
  "holdout_fraction": 0.4,
  "jobs": 4,
  "selection_metric": "macro_f1"
}
```

#### Step 3: Lock the Selection (Run ONCE)
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline lock-final \
  --config config/E7_final_evaluation.json \
  --data-dir data \
  --mapping config/feature_mapping.json \
  --reason "Selected based on E4 macro_f1_mean=0.842 (2018, weighted, K=50, alpha=0.5, seed=42) across 3 seeds"
```

**Output**: Creates `config/E7_final_evaluation.lock.json` — **immutable record**.

#### Step 4: Run E7 (Reads SUPPLEMENTARY)
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E7 \
  --config config/E7_final_evaluation.json \
  --data-dir data \
  --results-dir results \
  --mapping config/feature_mapping.json
```

**Output**: `results/current/E7.csv` — **Single-row result on SUPPLEMENTARY test set**.

> **Guard**: If you try to run E7 again with same config/data/code, it will fail with:
> `ValueError: SUPPLEMENTARY evaluation already attempted for this dataset; review its audit record`

---

## 5. Running Semantic Mapping Validation

The new `validate_mapping.py` script validates the 77→7 feature mapping assumption.

### 5.1 Basic Usage (Default Thresholds)
```bash
!source .venv/bin/activate && python -m ids_xgb.validate_mapping \
  --data-dir data \
  --mapping config/feature_mapping.json \
  --output results/mapping_validation.json
```

### 5.2 Custom Thresholds
```bash
!source .venv/bin/activate && python -m ids_xgb.validate_mapping \
  --data-dir data \
  --mapping config/feature_mapping.json \
  --output results/mapping_validation.json \
  --ks-threshold 0.05 \
  --wasserstein-threshold 0.10 \
  --mean-ratio-threshold 2.0 \
  --zero-fraction-threshold 0.50 \
  --group seven
```

### 5.3 Interpret Results
```bash
!source .venv/bin/activate && python -c "
import json
with open('results/mapping_validation.json') as f:
    r = json.load(f)
print('OVERALL:', 'PASS' if r['overall_pass'] else 'FAIL')
for feat, res in r['features'].items():
    status = '✓' if res['pass'] else '✗'
    print(f'  {status} {feat}: KS={res[\"ks_pvalue\"]:.4f}, W={res[\"wasserstein_distance\"]:.4f}, ratio={res[\"mean_ratio\"]:.2f}, zero_frac={res[\"zero_fraction\"]:.2f}')
"
```

**Thresholds (configurable):**
| Metric | Default | Meaning |
|--------|---------|---------|
| KS p-value | 0.05 | Distribution similarity (higher = more similar) |
| Wasserstein distance | 0.10 | Earth mover's distance (lower = more similar) |
| Mean ratio | 2.0 | Ratio of means (closer to 1 = more similar) |
| Zero fraction | 0.50 | Fraction of zeros (flag if > 50% in either domain) |

---

## 6. Viewing & Analyzing Results

### 6.1 List All Runs
```bash
!source .venv/bin/activate && python -m ids_xgb.pipeline list --results-dir results
```

### 6.2 Load Results for Analysis
```python
import pandas as pd

# Load all stage results
stages = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
results = {}
for s in stages:
    path = f'results/current/{s}.csv'
    try:
        results[s] = pd.read_csv(path)
        print(f'{s}: {len(results[s])} rows')
    except FileNotFoundError:
        print(f'{s}: not found')

# Example: Compare few-shot methods on E4
if 'E4' in results:
    e4 = results['E4']
    print(e4.groupby(['method', 'k_per_class'])['macro_f1_mean'].mean().unstack())
```

### 6.3 Export Results to Drive
```bash
!mkdir -p /content/drive/MyDrive/ids_results
!cp -r results /content/drive/MyDrive/ids_results/
```

---

## 7. Troubleshooting

### 7.1 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: data/ciciot2023_supplementary_test.csv` | Missing SUPPLEMENTARY file | Upload all 4 CSV files to `data/` |
| `ValueError: SUPPLEMENTARY evaluation already attempted` | E7 already run for this config | Check `results/runs/*/selection_lock.json`; use new config or new results-dir |
| `ValueError: E7 configuration, code, mapping or training data changed since lock-final` | Modified config/code/data after lock | Re-run `lock-final` with current state, then run E7 |
| `ModuleNotFoundError: No module named 'ids_xgb'` | Package not installed | Run `pip install -e .` in `.venv` |
| `CUDA out of memory` | GPU memory exhausted | Reduce `n_estimators` in config or use CPU: `export CUDA_VISIBLE_DEVICES=""` |

### 7.2 Reset E7 Lock (If Needed)
```bash
# Remove lock file to re-run E7 with same config
!rm config/E7_final_evaluation.lock.json
# Remove evaluation claim file
!rm data/.final_evaluation_*.json
```

### 7.3 Run on CPU Only
```bash
!export CUDA_VISIBLE_DEVICES="" && source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E4 ...
```

---

## 8. Complete Command Summary (Copy-Paste)

```bash
# ===== SETUP =====
!apt-get update -qq && apt-get install -y -qq git python3.12-venv 2>/dev/null | tail -5
!git clone https://github.com/YOUR_USERNAME/ids-xgboost-cross-domain.git
%cd ids-xgboost-cross-domain
!python3.12 -m venv .venv
!source .venv/bin/activate && pip install --upgrade pip setuptools wheel
!source .venv/bin/activate && pip install -e ".[dev]"

# ===== DATA =====
# Upload your 4 CSV files to data/ directory
!mkdir -p data
# (Use Colab file upload or Google Drive mount)
!ls -la data/

# ===== VALIDATE MAPPING =====
!source .venv/bin/activate && python -m ids_xgb.validate_mapping \
  --data-dir data --mapping config/feature_mapping.json \
  --output results/mapping_validation.json

# ===== PIPELINE E0-E6 =====
for stage in E0 E1 E2 E3 E4 E5 E6; do
  echo "=== Running $stage ==="
  !source .venv/bin/activate && python -m ids_xgb.pipeline run --stage $stage \
    --data-dir data --results-dir results --mapping config/feature_mapping.json
done

# ===== E7: SELECT BEST CONFIG THEN LOCK & RUN =====
# 1. Inspect results to pick best config
# 2. Edit config/E7_final_evaluation.json with your selection
# 3. Lock it
!source .venv/bin/activate && python -m ids_xgb.pipeline lock-final \
  --config config/E7_final_evaluation.json \
  --data-dir data --mapping config/feature_mapping.json \
  --reason "Your selection reason here"

# 4. Run E7 (reads SUPPLEMENTARY)
!source .venv/bin/activate && python -m ids_xgb.pipeline run --stage E7 \
  --config config/E7_final_evaluation.json \
  --data-dir data --results-dir results --mapping config/feature_mapping.json

# ===== EXPORT =====
!cp -r results /content/drive/MyDrive/ids_results_$(date +%Y%m%d)
```

---

## 9. Important Reminders

1. **Terminology**: All code/docs now use **SUPPLEMENTARY** (not FINAL) — see `README.md` disclaimers
2. **Data Snooping**: The SUPPLEMENTARY set was studied in a prior project — results are **supplementary**, not independent confirmation
3. **Lock-Final**: E7 lock is **immutable** — choose carefully, document reason
4. **Validation**: Run `validate_mapping.py` before trusting cross-domain results
5. **Transductive Methods**: E8 TTA (self-training, entropy) are transductive — marked in outputs
6. **Calibration Budget**: If using calibration (E8), 30% of K is reserved for sigmoid fitting

---

## 10. Support

- Check `README.md` for full project documentation
- Run `python -m ids_xgb --help` for CLI reference
- Run `python -m ids_xgb.pipeline --help` for pipeline commands
- Run `python -m ids_xgb.validate_mapping --help` for validation options