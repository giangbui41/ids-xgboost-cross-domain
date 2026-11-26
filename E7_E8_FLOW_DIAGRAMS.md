# 📊 Đồ thị luồng E7 / E8 — Phiên bản chính xác theo code

> ✅ **Đã验证 với source code:** `fewshot.py`, `improvements.py`, `adaptation.py`, `pipeline.py`, config `E7_final_evaluation.json`, `E8_improvements.json`

---

## 1. Luồng dữ liệu E7 (Few-shot) & E8 (Improvements)

```mermaid
flowchart TD
    subgraph DATA["📁 DỮ LIỆU"]
        SRC[("Source Domain<br/>CIC-IDS 2017 / 2018<br/>đầy đủ nhãn")]
        TGT[("Target Domain<br/>CICIoT2023 DEV<br/>ít nhãn")]
    end

    subgraph SPLIT["✂️ CHIA DỮ LIỆU (prepare_fewshot)"]
        direction TB
        S1[Source: train/val/test split<br/>stratified, seed=42,43,44]
        S2[Target DEV: 60% adaptation pool<br/>40% holdout evaluation pool]
        S3[Adaptation pool: K labeled/class + unlabeled]
    end

    subgraph FE["🔧 FEATURE ENGINEERING (chỉ E8)"]
        FE_OFF{{"feature_engineering: false<br/>(E7, một số variant E8)"}}
        FE_ON{{"feature_engineering: true<br/>(E8 variants: weighted_fe,<br/>adaptive_features, ssl_features,<br/>dann_xgb)"}}
        FEAT[FeatureEngineer:<br/>• Ratios: fwd/bwd pkts, bytes<br/>• Interactions: std×rate, range×rate<br/>• Log transforms heavy-tailed]
    end

    subgraph METHOD["⚙️ CHỌN PHƯƠNG PHÁP & TẠO TRỌNG SỐ"]
        M1["① weighted<br/>source×(1-α) + target×α<br/>α=0.5 default<br/>KHÔNG dùng unlabeled"]
        M2["② adaptive / weighted_adaptive<br/>KDE density HOẶC uncertainty entropy<br/>trên unlabeled (xu)<br/>→ nhân vào trọng số source"]
        M3["③ self_training<br/>Pseudo-label unlabeled (conf≥0.95)<br/>Retrain lặp (iter=3, pw=0.2)<br/>Có thể + adaptive_weighting"]
        M4["④ hybrid (E8 only)<br/>DANN encoder → frozen → XGBoost<br/>Gradient reversal domain adversarial<br/>Unlabeled dùng cho DANN"]
    end

    subgraph TRAIN["🏋️ HUẤN LUYỆN"]
        TR_XGB[XGBoost<br/>sample_weight từ phương pháp trên<br/>max_depth=3, lr=0.05, n_est=200]
        TR_DANN[DANN+XGBoost (hybrid)<br/>epochs=10, batch=256, λ=0.1]
        CAL[Calibration (variant calibrate=true)<br/>Platt sigmoid / Isotonic<br/>Fit trên calibration set tách từ K]
    end

    subgraph THRESH["🎯 CHỌN NGƯỠNG (KHÔNG dùng holdout)"]
        TH_FIXED["Fixed: 0.5"]
        TH_COST["Budget validation cost:<br/>Minimize FP×100 + FN×1<br/>Trên source validation HOẶC<br/>target calibration set (tách từ K)"]
    end

    subgraph EVAL["📊 ĐÁNH GIÁ TRÊN HOLDOUT (chỉ đo, không học)"]
        EV_IND["independent_holdout<br/>(mặc định: weighted, adaptive,<br/>self_training không TTA)"]
        EV_TRANS["transductive_holdout<br/>(chỉ variant có TTA:<br/>entropy / self_training TTA)"]
        METRICS[Macro-F1, Balanced Acc, AUROC, AUPRC<br/>FPR, FNR, Total Cost, Cost/sample<br/>ECE, Brier, LogLoss]
    end

    subgraph OUT["💾 OUTPUT"]
        O1[summary.csv + aggregate.csv]
        O2[manifest.json (SHA256 mọi artifact)]
        O3[model.json / calibrated_model.joblib / hybrid_model/]
        O4[predictions.npz, explanations/, run.json]
    end

    SRC --> S1
    TGT --> S2
    S2 --> S3
    S1 --> FE
    S3 --> FE
    FE_OFF --> METHOD
    FE_ON --> FEAT --> METHOD
    METHOD --> M1 & M2 & M3 & M4
    M1 --> TR_XGB
    M2 --> TR_XGB
    M3 --> TR_XGB
    M4 --> TR_DANN
    TR_XGB --> CAL
    TR_DANN --> CAL
    CAL --> TH_FIXED & TH_COST
    TH_FIXED --> EV_IND & EV_TRANS
    TH_COST --> EV_IND & EV_TRANS
    EV_IND --> METRICS
    EV_TRANS --> METRICS
    METRICS --> OUT

    classDef data fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef split fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef fe fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef method fill:#e8eaf6,stroke:#283593,stroke-width:2px;
    classDef train fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
    classDef thresh fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    classDef eval fill:#fff8e1,stroke:#f57f17,stroke-width:2px;
    classDef out fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    class SRC,TGT data;
    class S1,S2,S3 split;
    class FE_OFF,FE_ON,FEAT fe;
    class M1,M2,M3,M4 method;
    class TR_XGB,TR_DANN,CAL train;
    class TH_FIXED,TH_COST thresh;
    class EV_IND,EV_TRANS,METRICS eval;
    class O1,O2,O3,O4 out;
```

---

## 2. Ba phương pháp tạo trọng số (Weighting Methods)

```mermaid
flowchart TD
    subgraph INPUT["📥 INPUT"]
        XS[Source features Xs]
        YS[Source labels Ys]
        XT[Target labeled features Xt]
        YT[Target labeled labels Yt]
        XU[Unlabeled target Xu<br/>(chỉ adaptive & self_training)]
        ALPHA[α = 0.5<br/>(config)]
    end

    subgraph W1["① WEIGHTED (cơ bản)"]
        W1A[Balanced class weight trên Ys] --> W1B[Nhân (1-α)]
        W1C[Balanced class weight trên Yt] --> W1D[Nhân α]
        W1B & W1D --> W1E[Ghép nối → sample_weight<br/>KHÔNG dùng XU]
    end

    subgraph W2["② ADAPTIVE / WEIGHTED_ADAPTIVE"]
        W2A[Xs, Ys, Xt, Yt, XU] --> W2B["KDE Density Strategy<br/>log p(x|target) - log p(x|source)<br/>→ weight = exp(score)"]
        W2A --> W2C["Uncertainty Strategy<br/>Entropy predictions trên XU<br/>→ weight = 1 / (1 + entropy)"]
        W2B & W2C --> W2D["Clip [min_weight, max_weight]<br/>default [0.1, 10]<br/>Nhân vào source weight"]
    end

    subgraph W3["③ SELF_TRAINING"]
        W3A[Factory: XGBoost mới mỗi iter] --> W3B["Iter 1..3:<br/>• Fit trên labeled (Xs+Xt, Ys+Yt)<br/>• Predict proba trên XU<br/>• Chọn conf ≥ 0.95<br/>• Gán pseudo-label<br/>• Thêm vào train set với pseudo_weight=0.2"]
        W3B --> W3C[Model cuối cùng]
    end

    subgraph W4["④ HYBRID DANN+XGB (E8 only)"]
        W4A[DANN Encoder<br/>Gradient Reversal Layer<br/>Domain classifier: max confusion] --> W4B[Frozen encoder → features] --> W4C[XGBoost trên encoded features<br/>sample_weight từ weighted/adaptive]
    end

    INPUT --> W1 & W2 & W3 & W4

    classDef input fill:#e3f2fd,stroke:#1565c0;
    classDef w1 fill:#e8eaf6,stroke:#283593;
    classDef w2 fill:#fce4ec,stroke:#c2185b;
    classDef w3 fill:#e0f2f1,stroke:#00695c;
    classDef w4 fill:#fff3e0,stroke:#ef6c00;

    class XS,YS,XT,YT,XU,ALPHA input;
    class W1A,W1B,W1C,W1D,W1E w1;
    class W2A,W2B,W2C,W2D w2;
    class W3A,W3B,W3C w3;
    class W4A,W4B,W4C w4;
```

---

## 3. So sánh nhanh: E7 vs E8

| Đặc điểm | E7 (Few-shot) | E8 (Improvements) |
|----------|---------------|-------------------|
| Config | `E7_final_evaluation.json` | `E8_improvements.json`, `E8_hybrid.json`, `E8_tta.json` |
| K per class | 0,1,2,5,10,20,50,100 | 10, 50, 200 |
| Feature Engineering | ❌ Không | ✅ Có (per-variant toggle) |
| Calibration | ❌ Không | ✅ Platt/Isotonic (variant calibrate) |
| TTA (Test-Time Adapt) | ❌ Không | ✅ Entropy + Self-training TTA |
| Hybrid DANN+XGB | ❌ Không | ✅ Variant `dann_xgboost` |
| Seeds | 42, 43, 44 | 42, 43, 44 |
| Output | Learning curve K vs F1/Cost | Best model để `freeze` → `evaluate-final` |

---

## 4. Protocol quan trọng (nhắc nhở cho slide)

> ✅ **TUÂN THỦ NGHIÊM NGẶT:**
> - `prepare-iot` tạo DEV / SUPPLEMENTARY_TEST **tách biệt hoàn toàn**
> - E0–E8 **CHỈ đọc DEV**, không bao giờ đọc SUPPLEMENTARY_TEST
> - Ngưỡng (threshold) fix trên source validation HOẶC calibration set tách từ K — **KHÔNG retune trên holdout**
> - Imputation: **chỉ train medians** áp cho val/test/holdout
> - `freeze` khóa model → `evaluate-final` chạy **DUY NHẤT 1 LẦN** trên SUPPLEMENTARY_TEST
> - Mọi artifact có SHA256: data, mapping, config, code

---

## 5. Các lỗi trong bản vẽ cũ (để bạn biện minh nếu thầy hỏi)

| Sai trong bản cũ | Đúng theo code |
|------------------|----------------|
| "Đã tắt feature engineering" | E8 có variant bật FE (`feature_engineering: true`) |
| Holdout dùng chọn ngưỡng | Ngưỡng chọn trên source validation HOẶC calibration set tách từ K |
| Unlabeled dùng cho mọi phương pháp | **Weighted KHÔNG dùng unlabeled**; chỉ adaptive & self_training dùng |
| Mọi self-training = transductive | Chỉ variant có `tta != "none"` mới transductive; self_training thường = independent_holdout |

---

## 6. Cách nhúng vào slide

**Option A: Mermaid (nếu tool hỗ trợ)**
- Copy block mermaid đầu tiên vào slide markdown/notion/obsidian/hackmd

**Option B: Export PNG từ Mermaid Live Editor**
1. Mở https://mermaid.live
2. Paste code mermaid
3. Export PNG → chèn vào PowerPoint

**Option C: Dùng file markdown này**
- File này đã lưu tại: `E7_E8_FLOW_DIAGRAMS.md`
- Mở trong VS Code + extension "Markdown Preview Mermaid Support" để xem render