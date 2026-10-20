# E8: bảy hướng mở rộng có kiểm soát

E8 là thí nghiệm phát triển trên IoT DEV. Không có cam kết tăng F1 2–10% hoặc hybrid luôn tốt nhất. Metric chính là `total_cost = 100 * FP + FN`, càng thấp càng tốt. Macro-F1, AUROC, AUPRC, FPR/FNR vẫn được lưu để nhìn thấy đánh đổi. Chi phí 100/1 là giả định của người làm thí nghiệm, chưa được xác thực bằng chi phí vận hành thực tế.

Từ bản 0.4.0, E4 mặc định dùng chung runner cải tiến với E8; xem [tích hợp E4](INTEGRATION_STATUS.md). Cấu hình E4 cũ không có `feature_engineering`, E1–E3 và E5–E7 giữ cách huấn luyện cũ. **E7 hiện chỉ hỗ trợ các phương pháp cũ; chưa dùng E7 để đánh giá model E4 cải tiến/E8.** Khi cần đánh giá các mô hình này trên FINAL, phải thiết kế và khóa giao thức riêng trước. E4/E8 không đọc file FINAL.

## Chạy

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"

# Kiểm tra nhanh trên dữ liệu DEV thật, giới hạn số mẫu huấn luyện.
.\.venv\Scripts\python.exe scripts/E8_improvements.py --config config/E8_smoke.json --data-dir data/processed

# Ma trận chính: hai nguồn, ba seed, K=10/50/200.
.\.venv\Scripts\python.exe scripts/E8_improvements.py --data-dir data/processed

# DANN cần PyTorch; chỉ cài extra khi dùng hybrid.
.\.venv\Scripts\python.exe -m pip install -e ".[hybrid]"
.\.venv\Scripts\python.exe scripts/E8_improvements.py --config config/E8_hybrid.json --data-dir data/processed

# TTA là nghiên cứu transductive riêng.
.\.venv\Scripts\python.exe scripts/E8_improvements.py --config config/E8_tta.json --data-dir data/processed

.\.venv\Scripts\python.exe -m pytest -q
```

Mỗi lệnh tạo snapshot mới và cập nhật `results/current/E8.csv`. Đọc `config.json` trong snapshot để phân biệt smoke, ma trận chính, hybrid và TTA; `current/E8.csv` chỉ là lần chạy mới nhất. `aggregate.csv` thống kê mean/std theo seed, tách variant và vai trò đánh giá. Không chọn ngưỡng hay tham số bằng holdout rồi gọi kết quả đó là kiểm định độc lập.

## 1. Feature engineering

`features.py` giữ các đặc trưng gốc, thêm log1p, tỷ lệ hướng truyền, tỷ lệ thống kê độ dài, độ phân tán tương đối và tương tác rate × length. Đây là tương tác số học; không khẳng định nó chính là byte/s khi định nghĩa phép đo giữa hai dataset chưa tương đương. Chia cho 0 được xử lý rõ ràng: 0/0 thành 0, tỷ lệ không xác định thành missing. Median imputation chỉ fit nguồn, hoặc nhãn target train với `target_only`.

So sánh `weighted_raw` và `weighted_features` trên cùng split, K và tham số cây. Có thể đổi `group` thành `without_backward` hoặc `length_only` trong bản sao config để kiểm tra mapping.

Temporal có API `causal_temporal_features(frame, metadata, time_column, group_columns, window=5)`. Nó dùng timestamp thật, tách nhóm thiết bị/phiên và chỉ lấy các sự kiện trước đó. Timestamp trùng trong nhóm bị từ chối vì không rõ thứ tự. CSV hiện tại không giữ metadata này nên E8 **không bật temporal**. Không thay timestamp bằng chỉ số dòng, không ghép lịch sử từ train và holdout ngẫu nhiên. Khi thu thập lại metadata, cần giao thức chia theo nhóm/thời gian trước khi tích hợp temporal vào E8. `improvement_examples.py` minh họa API bằng dữ liệu tổng hợp có timestamp thật trong ví dụ.

## 2. Chi phí và ngưỡng

`costs: {"fp": 100, "fn": 1}` thay đổi cách tính chi phí. E8 ghi cả `cost_per_sample`, `always_benign_cost`, `always_attack_cost`. Nếu mô hình không tốt hơn hai chính sách đơn giản này thì phải báo cáo rõ.

Mặc định ngưỡng 0,5 được khai báo trước, để ablation cơ chế học dễ so sánh. Để chọn ngưỡng bằng chi phí, sao chép config và dùng:

```json
"threshold": {
  "strategy": "budget_validation_cost",
  "calibration_fraction": 0.3
}
```

Với K=10, cách này dành 3 nhãn mỗi lớp làm calibration, 7 nhãn mỗi lớp để train. Tổng vẫn là 20 nhãn; calibration không nằm trong train hoặc unlabeled pool. `source_only` dùng source validation. Tìm ngưỡng có xét cả chính sách không cảnh báo và ưu tiên ít cảnh báo khi bằng chi phí. K quá nhỏ bị từ chối.

Tập K-per-class có tỷ lệ lớp 50/50; ngưỡng tối thiểu hóa chi phí trên đó không nhất thiết tối ưu ở tỷ lệ lớp triển khai. Không dùng công thức ngưỡng Bayes như một bảo đảm khi score chưa được hiệu chỉnh xác suất. Cân lớp/trọng số khi train cũng không đồng nghĩa mô hình đã tối thiểu hóa chi phí 100/1.

## 3. TreeSHAP

Không cần cài thêm `shap`: `explanations.py` dùng TreeSHAP gốc của XGBoost và kiểm tra tổng contribution + bias bằng raw margin. Artifact mỗi variant có:

- `explanations/explanations.html`: báo cáo tương tác mở offline, so source/target và từng quyết định.
- `domain_comparison.csv`: trung bình SHAP có dấu và trị tuyệt đối ở hai miền.
- `source_contributions.csv`, `target_contributions.csv`: giá trị đầu vào, contribution, score và bias.
- `dataset_row_ids.json`: truy ngược mẫu về CSV gốc.
- `decision_policy.json`: ngưỡng và vai trò TTA.

Chênh lệch contribution là dấu hiệu để điều tra, không chứng minh thiên lệch nhân quả hay mapping sai. Với hybrid, SHAP giải thích các chiều embedding của XGBoost head, không phải trực tiếp feature mạng ban đầu. Với entropy TTA, TreeSHAP giải thích score trước phép biến đổi nhiệt độ; nhiệt độ được lưu trong `run.json`.

## 4. Adaptive instance weighting

`method: "adaptive"` mặc định dùng logistic discriminator nguồn/đích để ước lượng tỷ số mật độ, giới hạn multiplier trong [0.1, 10] và chuẩn hóa. Có thể dùng `adaptive.strategy: "uncertainty"` để thử tín hiệu độ tự tin của mô hình từ nhãn target được cấp.

Mô hình target từ nhãn ít ỏi cũng được dùng tìm mẫu nguồn có dự đoán tự tin trái nhãn nguồn, rồi giảm trọng số. `conflict_fraction` là **heuristic**, không phải chứng minh hai miền xung đột. Ghi cả effective source rows, min/max trọng số và tổng trọng số mỗi miền. Density discriminator fit tối đa 20.000 mẫu mỗi miền để giới hạn chi phí; sau đó tính multiplier trên toàn bộ nguồn train được chọn.

## 5. Test-time adaptation — tùy chọn

`E8_tta.json` dùng cùng dữ liệu để so không TTA, entropy và self-training TTA. Có score/metric trước TTA, cùng ngưỡng đã cố định. Mọi hàng TTA được gắn `evaluation_role=transductive_holdout`: **đặc trưng holdout đã tham gia thích nghi**, dù nhãn chưa được dùng.

Entropy ở đây tối ưu một tham số temperature của score, không lấy gradient xuyên qua cây. Temperature dương giữ thứ hạng và quyết định tại 0,5; không kỳ vọng cải thiện F1 tại ngưỡng này. Với ngưỡng khác, nó thay đổi điểm hoạt động và có thể khuếch đại sai lầm.

Self-training TTA huấn luyện lại từ nhãn thật với nhãn giả của holdout. Nếu phương pháp gốc là SSL, lần này dùng chung adaptation + holdout không nhãn; được ghi `joint_transductive_retraining`, không gọi là tiếp tục model SSL cũ. Không gộp kết quả này với đánh giá giữ toàn bộ holdout ngoài quá trình học.

## 6. Semi-supervised learning

`method: "self_training"` dùng toàn bộ phần adaptation 60% trừ mẫu K-per-class đã chọn. Chỉ đặc trưng của phần còn lại được đưa vào hàm học; nhãn thật của chúng không được truyền. Có thể giới hạn `max_unlabeled_samples` để smoke; giới hạn phải được công bố khi báo cáo.

Mỗi vòng chọn nhãn giả đủ tự tin, giới hạn theo lớp, không chọn lặp lại một dòng. Model được fit lại từ nhãn thật + nhãn giả tích lũy. Tổng trọng số nhãn giả không vượt `pseudo_weight × tổng trọng số nhãn thật`. Không có mẫu đủ tự tin thì dừng. Confidence cao chưa bảo đảm nhãn giả đúng dưới domain shift; kiểm tra log số mẫu/lớp và so với baseline không SSL.

## 7. Hybrid DANN + XGBoost

Đây là phần **tùy chọn** được nêu trong checklist Week 3+ của `INTEGRATION_GUIDE.md`: “Hybrid DANN+XGBoost (nếu cần)”. E4 mặc định không chạy hybrid, không cần PyTorch.

`hybrid.py` dùng PyTorch MLP encoder, task head và domain head với gradient reversal. Scaler chỉ fit nguồn. Task loss dùng nhãn nguồn và nhãn target trong ngân sách; domain loss dùng danh tính miền. Encoder huấn luyện số epoch định trước, rồi đóng băng để XGBoost học embedding. Đây là biến thể có ít nhãn target, không trùng hoàn toàn với DANN chỉ có nguồn có nhãn.

Chạy thêm bản sao `E8_hybrid.json` với `hybrid.lambda_domain=0` để kiểm soát tác động của domain adversarial loss. Giữ các tham số khác như cũ. Hybrid không được giả định tốt hơn XGBoost raw; so cả total_cost, F1, thời gian và ngân sách dữ liệu.

Artifact gồm encoder state dict, scaler JSON, XGBoost JSON và cấu hình/lịch sử; có `HybridDANNXGB.load()`. Default hybrid cap 20.000 mẫu unlabeled được ghi trong artifact.

## Calibration xác suất (bổ sung 0.4.0)

`calibration.py` cung cấp `calibrate_model(..., cv=5)`, `calibrate_prefit()` và `check_calibration()`. Trong E8, thêm `calibrate: true` vào một variant để fit sigmoid (hoặc isotonic) trên nhãn riêng trong ngân sách target; source-only dùng source validation. `calibration: {"method": "sigmoid", "fraction": 0.3, "n_bins": 10}` cấu hình phép hiệu chỉnh. Nếu đồng thời chọn ngưỡng theo chi phí, cùng phần nhãn được dành riêng theo `threshold.calibration_fraction` dùng cho calibration và chọn ngưỡng.

`calibration.json` ghi ECE, Brier score, log loss và reliability bins trên holdout trước/sau. Holdout chỉ dùng báo cáo, không dùng quyết định có calibration hay chọn phương pháp. `calibrated_model.joblib` chứa mô hình hoàn chỉnh; `model.json` chỉ là cây gốc. TreeSHAP giải thích cây trước calibration. Các variant calibration và TTA/hybrid được chạy riêng. ECE trên ít nhãn có sai số lớn; tập calibration K-per-class cân bằng không có cùng prevalence với dữ liệu triển khai.

## Giới hạn bằng chứng

Smoke và kiểm thử xác nhận chương trình chạy đúng, không xác nhận mức tăng F1. Ba seed và split ngẫu nhiên cũng chưa chứng minh độc lập theo phiên/thiết bị. Cần giữ kết quả bất lợi, báo rõ mọi lần chọn cấu hình dựa trên DEV, và đánh giá bằng một giao thức dữ liệu mới khi muốn đưa ra kết luận nghiên cứu mạnh hơn.

Tham chiếu triển khai: [XGBoost prediction/TreeSHAP](https://xgboost.readthedocs.io/en/latest/prediction.html), [DANN gốc](https://jmlr.org/papers/v17/15-239.html), [self-training và độ tin cậy xác suất](https://scikit-learn.org/1.5/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html).
