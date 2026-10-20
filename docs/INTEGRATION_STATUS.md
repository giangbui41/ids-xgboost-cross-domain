# Tích hợp INTEGRATION_GUIDE.md — phiên bản 0.4.0

Bảy bước đã có đường chạy trong E4. Các đoạn mã trong hướng dẫn là ví dụ; triển khai dùng chung các module E4/E8 để giữ split, ngân sách nhãn và artifact nhất quán.

| Bước | Triển khai |
|---|---|
| Feature engineering | `features.engineer_features`, `prepare.load_and_engineer_features`; E4 mặc định bật `FeatureEngineer` |
| Cost-sensitive metrics | `experiment.metrics_with_cost`/`metrics_cost_sensitive`; giữ API `metrics`, chi phí mặc định FP=100/FN=1 |
| SHAP | `explanations.export_tree_explanations`; TreeSHAP gốc XGBoost, xuất HTML/CSV, không cần cài thêm `shap` |
| Adaptive weighting | `fewshot.run_fewshot_improvements` gọi runner chung; E4 nhận `weighted_adaptive`, mặc định uncertainty, có density |
| TTA | `use_tta: true`, `tta_method: entropy` hoặc `self_training`; thêm hàng đối chiếu với weighted không TTA |
| Calibration | `calibration.calibrate_model`, `calibrate_prefit`, `check_calibration`; E4 có variant `weighted_calibrated` |
| Semi-supervised | `semisupervised.fit_self_training`; runner dùng cùng lõi `adaptation.self_train` |

`improvement_examples.py` có API tham khảo và ví dụ dữ liệu tổng hợp. Không import script ở thư mục gốc từ package. `pyproject.toml` và `ids_xgb.__version__` cùng là 0.4.0. `*.egg-info` là metadata sinh khi cài package, không phải nơi chỉnh thuật toán.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/E0_data_audit.py --data-dir data/processed
.\.venv\Scripts\python.exe scripts/E4_fewshot.py --config config/E4_fewshot_smoke.json --data-dir data/processed
# Ma trận chính: hai nguồn, ba seed, K=10/50/200, không giới hạn nguồn/pool không nhãn
.\.venv\Scripts\python.exe scripts/E4_fewshot.py --data-dir data/processed
# TTA tùy chọn; thay tta_method trong bản sao config để thử self_training
.\.venv\Scripts\python.exe scripts/E4_fewshot.py --config config/E4_tta.json --data-dir data/processed
```

Mỗi lần chạy có snapshot riêng trong `results/runs`, CSV hiện hành ở `results/current/E4.csv`. Snapshot chứa cấu hình, hash đầu vào/mã, split, dự đoán, mô hình và giải thích. Nếu đã có kết quả cũ, sao lưu thư mục kết quả trước khi chạy; snapshot cũ không bị ghi đè. Smoke chỉ có một seed, tối đa 4.000 mẫu nguồn và 4.000 mẫu không nhãn, 30 cây: dùng kiểm tra tích hợp, không thay thế ma trận nghiên cứu chính.

## Ngân sách, calibration và đánh giá

- E4 lấy K nhãn mỗi lớp từ adaptation pool; holdout độc lập với train/calibration. Nhãn calibration nằm trong 2K, bị loại khỏi cả train lẫn pool không nhãn.
- Variant `weighted_calibrated` dành `ceil(0.3*K)` nhãn mỗi lớp để fit sigmoid. Source-only nếu bật calibration trong E8 dùng source validation đã tách khỏi source train.
- `check_calibration` trả ECE, Brier score, log loss và reliability bins. `is_well_calibrated` chỉ là so ECE với tolerance, không phải chứng nhận xác suất đúng.
- Calibration được bật bằng cấu hình trước khi nhìn holdout. Báo cáo cả kết quả xấu đi; không dùng holdout để quyết định hiệu chỉnh. Tập calibration cân bằng có thể không đại diện tỷ lệ attack triển khai.
- Để dự đoán lại variant đã calibration, load `calibrated_model.joblib` bằng joblib, áp dụng FeatureEngineer và median trong `run.json`. `model.json` chỉ chứa cây trước calibration. TreeSHAP cũng giải thích cây gốc, không phải toàn bộ phép hiệu chỉnh.
- Entropy TTA điều chỉnh temperature, không lấy gradient qua cây. Nó giữ thứ hạng và quyết định tại ngưỡng 0,5; không hứa tăng F1/AUROC. Các hàng TTA ghi `transductive_holdout` do dùng đặc trưng holdout để thích nghi; nhãn holdout chỉ dùng tính metric.

## Hybrid và E7

Hướng dẫn gốc có mục checklist Week 3+: **“Hybrid DANN+XGBoost (nếu cần)”**. Vì vậy giữ code hybrid có sẵn như tùy chọn E8, không đưa vào E4 mặc định, không cần cài torch để chạy E4.

Không sửa `config/E7_final_evaluation.json` hoặc chạy FINAL trong đợt tích hợp này. E7 hiện chỉ tái huấn luyện giao thức cũ, không thể dùng nó để xác nhận kết quả FINAL của model cải tiến. Cần giao thức FINAL phù hợp được khóa trước một đợt đánh giá riêng.

Mức tăng 1–5% trong hướng dẫn là kỳ vọng, không phải tiêu chí có thể đảm bảo bằng sửa mã. Báo cáo so sánh cần dựa trên CSV thực tế, cùng split và ngân sách.
