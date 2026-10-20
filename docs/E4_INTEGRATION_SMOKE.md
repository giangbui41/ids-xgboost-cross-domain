# Kiểm tra tích hợp E4 0.4.0

Đã chạy `python -m pytest -q`: **98 passed**. Đã chạy E0 trước E4 trên bộ dữ liệu đang có; không chạy E7 hoặc đọc IoT FINAL. Đã sao lưu kết quả E0 vào `results.backup_before_E4_0.4.0` trước khi chạy E4.

- E0 run: `20260929T134557Z_E0_7dc6c237`.
- E4 run: `20260929T134632Z_E4_3836eed1`.
- Cấu hình: [`E4_fewshot_smoke.json`](../config/E4_fewshot_smoke.json).
- CSV: [`summary.csv`](../results/runs/20260929T134632Z_E4_3836eed1/summary.csv).
- Nguồn 2018 → IoT DEV, seed 42, K=10/lớp, holdout 125.532 dòng; tối đa 4.000 dòng nguồn train và 4.000 dòng không nhãn; 30 cây, ngưỡng 0,5, chi phí FP/FN=100/1. Source-only dùng K=0.

| Variant | Macro-F1 | AUROC | Total cost | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| source_only | 0.0228 | 0.3330 | 127165 | 0.7767 | 0.6295 |
| target_only | 0.6133 | 0.9582 | 31517 | 0.2934 | 0.1136 |
| pooled | 0.0227 | 0.6251 | 126969 | 0.7809 | 0.6338 |
| weighted | 0.5850 | 0.9504 | 31005 | 0.1850 | 0.1069 |
| weighted_adaptive | 0.5909 | 0.9514 | 30154 | 0.1891 | 0.1185 |
| self_training | 0.5947 | 0.9526 | 29692 | 0.1876 | 0.1072 |
| weighted_raw | 0.6102 | 0.9629 | 27364 | 0.1939 | 0.0948 |
| weighted_calibrated | 0.6155 | 0.9407 | 37392 | 0.1018 | 0.0984 |

Các variant dùng đặc trưng mới, trừ `weighted_raw`. Variant calibration dành 14 nhãn để huấn luyện cây và 6 nhãn để fit sigmoid; các variant target khác dùng đủ 20 nhãn train. Vì vậy không coi chênh lệch giữa `weighted` và `weighted_calibrated` là tác động riêng của sigmoid trên cùng cây.

So sánh đúng trước/sau trên **cùng cây của variant calibrated** được lưu trong [`calibration.json`](../results/runs/20260929T134632Z_E4_3836eed1/weighted_calibrated_2018_s42_k10/calibration.json):

| Metric | Trước | Sau |
|---|---:|---:|
| ECE | 0.20758 | 0.10177 |
| Brier | 0.14220 | 0.09837 |
| Log loss | 0.46707 | 0.61087 |

ECE/Brier giảm nhưng log loss tăng; cả hai ECE đều vượt tolerance 0,05. Không kết luận xác suất đã được hiệu chỉnh tốt trên miền đích.

Trong smoke này, đặc trưng mới không vượt weighted_raw về F1/AUROC/chi phí. Adaptive và SSL cải thiện một số chỉ số so với weighted có đặc trưng mới, nhưng chưa vượt weighted_raw về chi phí. Giữ nguyên toàn bộ kết quả này; không chọn lại cấu hình theo holdout trong đợt kiểm tra tích hợp.

Artifacts gồm model, dự đoán, split, nhãn đã chọn, HTML/CSV TreeSHAP và mô hình calibration có thể nạp lại. Kiểm thử tự động bao gồm TTA và tách ngân sách calibration; lượt smoke trên dữ liệu thật này không bật TTA/hybrid. Chưa chạy ma trận nghiên cứu đầy đủ hai nguồn × ba seed × ba ngân sách, chưa so sánh thực nghiệm DANN, chưa đánh giá FINAL. Vì vậy các số trên xác nhận đường chạy tích hợp, không chứng minh mức tăng 1–5% trong hướng dẫn.
