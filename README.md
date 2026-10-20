# Đồ án: IDS liên miền với XGBoost và ít nhãn IoT

**Bản 0.4.0 — tích hợp hướng dẫn vào E4:** feature engineering, chi phí `100 × FP + FN`, TreeSHAP, `weighted_adaptive`, self-training và calibration xác suất. TTA bật riêng. Xem [cách chạy và đối chiếu hướng dẫn](docs/INTEGRATION_STATUS.md). E8 vẫn cung cấp các ablation mở rộng; hybrid DANN + XGBoost là tùy chọn của checklist Week 3+, không thuộc cấu hình E4 mặc định.

Đây là mã nguồn để nghiên cứu câu hỏi: **Khi chỉ được gán nhãn rất ít lưu lượng IoT, kết hợp dữ liệu CIC-IDS2017/2018 với các nhãn đó có giúp XGBoost phát hiện tấn công trên IoT tốt hơn không?** Các đối chứng gốc là `source_only`, `target_only`, `pooled`, `weighted`; E4 mới thêm `weighted_adaptive`, `self_training`, so sánh đặc trưng gốc và calibration. Các cải tiến **chưa được mặc định coi là tốt hơn** các đối chứng.

Mã có config/script đánh số E0–E7 và tạo CSV theo phiên bản để khôi phục kết quả. Gói này không chứa CSV dữ liệu và không tự tải chúng.

## Chuẩn bị trên Windows



```text
data/processed/cicids2017_balanced_77feats.csv
data/processed/cleaned_data_sampled.csv
data/processed/ciciot2023_dev.csv
data/processed/ciciot2023_final_test.csv
```

Ba file đầu là đầu vào E0–E6. File `ciciot2023_final_test.csv` **không được các script E0–E6 đọc**. Bản 2017 đã cân bằng, 2018 đã làm sạch/lấy mẫu, và IoT DEV là mẫu 5%; ghi đúng xuất xứ này trong báo cáo, đừng gọi chúng là CSV thô.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Nếu PowerShell chặn kích hoạt, dùng `.\.venv\Scripts\python.exe` thay cho `python`. Khi cần kiểm thử, cài thêm bằng `python -m pip install -e ".[test]"`, rồi dùng `python -m pytest -q`. Các kiểm thử dùng dữ liệu giả trong thư mục tạm, không đọc FINAL thật.

## Kịch bản và lệnh chạy

Mỗi script đọc config cùng số trong `config/`. Các config E1–E6 mặc định chạy seed 42, 43, 44; có thể mất nhiều phút. Chạy theo thứ tự để hiểu dữ liệu trước khi diễn giải kết quả.

| Stage | Mục đích | Script và config |
|---|---|---|
| E0 | Kiểm toán 7 ánh xạ đặc trưng và tỷ lệ nhãn | `scripts/E0_data_audit.py`, `config/E0_data_audit.json` |
| E1 | Trong từng bộ 2017, 2018, IoT DEV; so ba mô hình | `scripts/E1_in_domain.py`, `config/E1_in_domain.json` |
| E2 | 2017→2018 và 2018→2017 | `scripts/E2_related_transfer.py`, `config/E2_related_transfer.json` |
| E3 | Zero-shot 2017→IoT và 2018→IoT | `scripts/E3_zero_shot_iot.py`, `config/E3_zero_shot_iot.json` |
| **E4** | **K nhãn/lớp IoT; đặc trưng mới, adaptive/SSL, calibration và đối chứng** | `scripts/E4_fewshot.py`, `config/E4_fewshot.json` |
| E5 | Độ nhạy của trọng số dữ liệu IoT `alpha` | `scripts/E5_alpha_sensitivity.py`, `config/E5_alpha_sensitivity.json` |
| E6 | Kiểm tra dịch chuyển phép đo trên các phương pháp E4 | `scripts/E6_measurement_drift.py`, `config/E6_measurement_drift.json` |
| E7 | Một lần đánh giá cấu hình đã chọn trên IoT FINAL | `scripts/E7_final.py`, `config/E7_final_evaluation.json` |

```powershell
python scripts/E0_data_audit.py --data-dir data/processed
python scripts/E1_in_domain.py --data-dir data/processed
python scripts/E2_related_transfer.py --data-dir data/processed
python scripts/E3_zero_shot_iot.py --data-dir data/processed
python scripts/E4_fewshot.py --data-dir data/processed
python scripts/E5_alpha_sensitivity.py --data-dir data/processed
python scripts/E6_measurement_drift.py --data-dir data/processed
```

E7 không nằm trong chuỗi lệnh trên. **E7 hiện chỉ hỗ trợ giao thức cũ: đặc trưng gốc và bốn phương pháp gốc. Không dùng E7 hiện tại để báo cáo FINAL cho E4 cải tiến hoặc E8.** Hướng dẫn khóa dưới đây chỉ áp dụng cho giao thức cũ. Khi chốt cấu hình cũ, **sao chép** `config/E7_final_evaluation.json` sang `config/E7_final_evaluation_locked.json`, ghi rõ `source`, `method`, `k_per_class`, `seed`, `model` và `alpha` nếu dùng weighted. Lưu lại lý do lựa chọn trước khi chạy:

```powershell
python -m ids_xgb.pipeline lock-final --config config/E7_final_evaluation_locked.json --data-dir data/processed --reason "Ghi ly do lua chon tu E4/E5 tai day"
python scripts/E7_final.py --config config/E7_final_evaluation_locked.json --data-dir data/processed
```

Config gốc để `UNSET` nên chạy trực tiếp sẽ báo lỗi. `lock-final` ghi file `.lock.json` cạnh config, chứa lý do chọn và hash của config, mã nguồn, ánh xạ, nguồn được chọn và IoT DEV; lệnh này không đọc FINAL. E7 kiểm tra các hash trước khi mở FINAL. Với `source_only`, đặt `k_per_class=0`.

E7 dùng các nhãn IoT DEV được chọn theo seed để huấn luyện rồi đánh giá trên IoT FINAL; kết quả ghi vào `results/current/E7.csv`. Trước mỗi lần thử, chương trình tạo file `.final_evaluation_<sha256>.json` cạnh dữ liệu. E7 và lệnh cũ `evaluate-final` dùng chung dấu vết này, nên đổi thư mục kết quả không cho phép đánh giá lại. Dấu vết được giữ cả khi lần chạy lỗi: cần kiểm tra nguyên nhân và lưu lịch sử xử lý, không tự động xóa để chạy lại. Đây là biện pháp ngăn chạy lặp ngoài ý muốn trong cùng thư mục dữ liệu, không chống việc cố ý xóa dấu vết hoặc sao chép dữ liệu sang nơi khác. FINAL từng được nghiên cứu ở đồ án trước, vì thế đây là tập đánh giá bổ sung có lịch sử đã nhìn thấy, không phải kiểm định độc lập hoàn toàn.

Có thể chạy riêng một script. Để chạy nhanh trước, **sao chép** config sang file mới, đổi `seeds` thành `[42]`, E4 đổi `sources` thành `["2018"]` và `k_per_class` thành `[10]`, rồi truyền `--config config/E4_fewshot_smoke.json`. Mỗi lần chạy tạo kết quả mới, không ghi đè snapshot cũ. Không sửa config sau khi chạy và coi đó là cùng một thí nghiệm; mỗi lần chạy lưu bản config riêng.

### Giao thức E4 chính

- Theo từng seed, chia phân tầng IoT DEV thành **60% adaptation pool, 40% development holdout**. Mọi phương pháp, nguồn và ngân sách cùng seed dùng cùng holdout. Đây là holdout *nội bộ của DEV*, không phải dữ liệu chưa từng được nhóm nghiên cứu nhìn thấy trong toàn bộ lịch sử dự án.
- Từ adaptation pool, chọn **K benign + K attack**, với K mặc định là 10, 50, 200. Mẫu của ngân sách nhỏ nằm trong ngân sách lớn. Người làm thí nghiệm biết nhãn để tạo benchmark phân tầng; mô hình chỉ nhận đúng 2K nhãn được chọn.
- `source_only`: không dùng nhãn IoT. `target_only`: huấn luyện bằng 2K nhãn IoT. `pooled`: gộp nguồn với 2K nhãn IoT theo trọng số lớp thông thường. `weighted`: cân lớp trong từng miền, rồi cấp tỷ lệ tổng trọng số `alpha` cho IoT (mặc định 0,5).
- XGBoost là mô hình E4 chính. Logistic Regression và Random Forest là đối chứng E1–E3. Có thể thay danh sách mô hình ở config khi mở rộng thí nghiệm.
- **Ngưỡng 0,5 cố định** trong cấu hình E4 chính và E5/E6. E4 có thể chọn ngưỡng theo chi phí trên nhãn validation thuộc ngân sách. Không dùng nhãn holdout để chọn ngưỡng hay tham số.
- E4 mặc định bật feature engineering và TreeSHAP. `weighted_raw` dùng đặc trưng gốc; `weighted_calibrated` dành 30% K mỗi lớp (làm tròn lên) để fit calibration sigmoid, phần còn lại huấn luyện cây. Tổng nhãn vẫn là 2K. Các hàng khác dùng đủ 2K nhãn để train. Đây là so sánh cùng ngân sách nhãn, không phải cùng số nhãn train.
- E4 và E8 dùng chung runner cải tiến. E5/E6 và cấu hình E4 cũ không có khóa `feature_engineering` vẫn dùng giao thức cũ. Không gộp hai giao thức thành một thí nghiệm.
- E5 báo cáo độ nhạy `alpha` chứ không chọn alpha theo holdout. E6 nhân các tốc độ và thống kê độ dài với `1±epsilon`; đây là stress test ở mức đặc trưng, **không phải FGSM/PGD hay tấn công gói tin có kiểm chứng**.

E1–E3 là bối cảnh và baseline; chúng không tái lập nguyên sáu mô hình, 14 đặc trưng E2 hoặc DANN của repo 3 tín. Đặc biệt E3 đánh giá trên toàn bộ IoT DEV, còn E4 so sánh các phương pháp trên holdout 40% riêng của DEV. **Dùng hàng `source_only` trong E4 để so sánh công bằng với các hàng E4 khác**, không lấy E3 đối chiếu trực tiếp.

`examples/E4_smoke_summary.csv` và `examples/E4_smoke_aggregate.csv` cho thấy định dạng kết quả của một lần chạy 2018→IoT, seed 42, K=10. Đây là kiểm tra mã, không phải kết luận nghiên cứu. Trong lần này `target_only` có Macro-F1 cao hơn `weighted`; phải giữ cả kết quả bất lợi khi báo cáo.

## CSV theo phiên bản và rollback

Mỗi lần chạy tạo một ID mới, ví dụ `20260927T094555Z_E4_23490ed1`:

```text
results/
  runs/<run-id>/summary.csv       CSV bất biến của lần chạy
  runs/<run-id>/aggregate.csv     trung bình/độ lệch theo seed
  runs/<run-id>/config.json        config đã dùng
  runs/<run-id>/manifest.json      hash CSV, mã và dữ liệu đầu vào
  runs/<run-id>/split_*.json      chỉ số chia nguồn/IoT cho E4–E6
  runs/<run-id>/selected_target_indices.json  nhãn IoT DEV đã chọn ở E7
  current/E4.csv                  bản E4 hiện được chọn
  registry.json                   danh sách các run và lịch sử chuyển bản
```

Xem ID và bản hiện hành:

```powershell
python -m ids_xgb.pipeline list
```

Khôi phục CSV hiện hành về run E4 trước đó:

```powershell
python -m ids_xgb.pipeline rollback --stage E4 --run-id 20260927T094555Z_E4_23490ed1
```

Rollback **kiểm tra SHA-256 và chỉ thay `results/current/E4.csv`**; mọi snapshot và `aggregate.csv` của từng run vẫn còn. Nó không đảo ngược mã nguồn, môi trường Python hay dữ liệu đầu vào. Các thao tác cập nhật registry dùng `.registry.lock` để từ chối hai tiến trình ghi cùng lúc. Nếu nhận lỗi registry đang bận, snapshot đã tạo vẫn còn nhưng có thể chưa được đăng ký/kích hoạt. Nên dùng `--results-dir` riêng cho các tiến trình. Nếu tiến trình bị dừng đột ngột, chỉ dọn khóa còn sót sau khi xác nhận không còn tiến trình ghi kết quả.

Đầu vào được kiểm tra và tính hash trước huấn luyện; chương trình chỉ yêu cầu những nguồn thực sự dùng trong config. Trước khi công bố snapshot, hash được kiểm tra lại để phát hiện dữ liệu hoặc mã thay đổi giữa chừng. Thư mục `.running_*` của lần lỗi được giữ để xem xét.

## Kiểm tra đặc trưng và trùng lặp

E2 ưu tiên các cặp tên trong `feature_mapping.json`, sau đó mới lấy giao của tên đã chuẩn hóa. Ví dụ `Flow_Packets/s` và `Flow Pkts/s` được ghép qua tên chuẩn `flow_packets_per_second`. Các đặc trưng chưa có ánh xạ chỉ được ghép nếu tên chuẩn hóa khớp hoàn toàn; chương trình không tự đoán mọi tên viết tắt. JSON của E2 ghi danh sách cột đầu vào, đặc trưng được chọn và số cột bị loại. Kết quả E2 mới cần chạy thành phiên bản riêng vì tập đặc trưng có thể khác bản trước.

E1 ghi mức trùng vector đặc trưng giữa train/validation và train/test trong JSON kết quả; E4–E6 ghi giữa adaptation pool/holdout trong `split_*.json`. Hai dòng có cùng đặc trưng chưa chắc là cùng sự kiện. Các báo cáo này không chứng minh độc lập theo phiên, thiết bị hoặc thời gian và không tự thay đổi split.

Khi tạo một bộ IoT mới bằng `prepare-iot`, có thể thêm `--deduplicate` để loại các dòng có toàn bộ đặc trưng số giống nhau trước khi chia DEV/FINAL. Nếu các vector giống nhau có nhãn mâu thuẫn, chương trình dừng để xử lý dữ liệu. Manifest ghi số dòng bị loại. Tùy chọn này làm thay đổi phân phối dữ liệu và không thay thế holdout theo thiết bị/thời gian; mặc định vẫn giữ dữ liệu như trước. Không ghi đè bộ dữ liệu đang dùng trong nghiên cứu.

## Tài liệu, giới hạn và bước báo cáo

- `config/feature_mapping.json` ghi 7 ánh xạ **ứng viên** giữa CICFlowMeter và CICIoT2023. Số liệu nhất quán không chứng minh chúng cùng ý nghĩa; `Drate` gần như toàn số 0 ở IoT DEV. Cần đối chiếu mã trích xuất/định nghĩa trước khi tuyên bố chuyển miền hợp lệ.
- Tỷ lệ attack trên IoT DEV khoảng 97,65%. AUPRC cao cần được so với mốc prevalence; FPR và số cảnh báo sai trên benign rất quan trọng.
- IoT DEV và bản FINAL đã được dùng trong đồ án cũ. Giữ FINAL ngoài E0–E6 ngăn thêm việc chọn mô hình theo nó, nhưng không biến nó thành tập độc lập hoàn toàn mới. Muốn kết luận mạnh hơn cần một tập dữ liệu đích mới hoặc holdout theo thiết bị/thời gian được tạo trước nghiên cứu.
- `weighted` hiện là một giả thuyết thực nghiệm đơn giản, không phải thuật toán mới đã được xác nhận. Nếu `target_only` thắng, phải báo cáo trung thực và phân tích khi nào dữ liệu nguồn gây hại.
- Chạy tối thiểu ba seed và báo cáo trung bình/độ lệch, đồ thị theo K, ablation nhóm đặc trưng, chi phí huấn luyện/suy luận; chỉ so DANN cũ khi dữ liệu, split, đặc trưng, ngưỡng và cách đo tương thích.

Nếu muốn dùng CSV chính thức thay đầu vào đã xử lý, mã `prepare-source`/`prepare-iot` trong `ids_xgb.cli` có thể tạo một bộ dữ liệu **mới**; không trộn chúng với CSV cũ trong một bảng. Trang phát hành: [2017](https://www.unb.ca/cic/datasets/ids-2017.html), [2018](https://www.unb.ca/cic/datasets/ids-2018.html), [IoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html). Bản IoT Kaggle đã được đối chiếu với protocol pool cũ là [himadri07/ciciot2023](https://www.kaggle.com/datasets/himadri07/ciciot2023).
