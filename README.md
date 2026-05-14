# FOMO Detection Pipeline

## Giới thiệu (Overview)
Dự án phát triển một hệ thống học máy (Machine Learning pipeline) để phát hiện hành vi FOMO (Fear Of Missing Out) của nhà đầu tư dựa trên dữ liệu giao dịch. Do thiếu nhãn thực tế (ground truth), dự án sử dụng phương pháp **Weak Supervision** với thư viện **Snorkel** để gán nhãn tự động (soft labels) dựa trên các tập luật/heuristics chuyên gia. Sau đó, một mô hình **XGBoost Regressor** được huấn luyện trên tập dữ liệu này để học cách dự đoán xác suất FOMO (`fomo_prob`) của bất kỳ giao dịch nào.

## Cấu trúc thư mục (Directory Structure)

Dự án bao gồm các thành phần chính sau:

- `fomo_pipeline_eda.ipynb`: File Jupyter Notebook tổng hợp, chứa toàn bộ quá trình Data Engineering, Feature Construction và Exploratory Data Analysis (EDA). File này cũng đã được đồng bộ bao gồm cả phần Validate Features và Train XGBoost ở phần cuối.
- `labeling_functions.py`: Định nghĩa các Labeling Functions (LFs) chứa các quy tắc/heuristics để xác định hành vi FOMO (ví dụ: RSI extreme, Value Spike...).
- `make_lf_input.py`: Script xử lý và trích xuất các features cần thiết để làm đầu vào cho các LFs.
- `run_snorkel.py`: Script chạy Snorkel LabelModel. Mô hình này tổng hợp các votes từ các LFs, giải quyết conflict/overlap và sinh ra nhãn mềm (`fomo_prob`) cho từng giao dịch.
- `validate_features.py`: Script kiểm tra tính hợp lệ của tập đặc trưng (features) trước khi đưa vào huấn luyện mô hình. Bao gồm các bước kiểm tra Sanity, Data Leakage (so với LF inputs), Feature Quality và Label Distribution.
- `train_xgboost_new.py`: Script huấn luyện mô hình XGBoost cuối cùng dựa trên các features đã được xác thực, kết hợp tối ưu hóa siêu tham số tự động bằng Optuna (TimeSeriesSplit CV).
- `input/`: Thư mục chứa dữ liệu gốc đầu vào.
- `output/`: Thư mục lưu trữ kết quả phân tích, dữ liệu trung gian (`lf_input.csv`, `fomo_features.csv`, `snorkel_labels.csv`), và lưu các mô hình (ví dụ mô hình XGBoost sau khi train).

## Pipeline các bước thực hiện (Pipeline Workflow)

Luồng thực thi chuẩn của hệ thống đi qua 5 bước chính:

1. **Data Prep & EDA**: Chạy các cell trong `fomo_pipeline_eda.ipynb` để xử lý dữ liệu giao dịch nguyên thủy, tính toán các metrics tài chính và xây dựng các đặc trưng (features) ban đầu.
2. **Chuẩn bị LF Input**: Chạy `make_lf_input.py` để tạo file input chuyên dụng cho Snorkel.
3. **Sinh nhãn (Weak Supervision)**: Chạy `run_snorkel.py` để áp dụng tập luật trong `labeling_functions.py` và tạo ra nhãn tự động cho tập training.
4. **Kiểm tra Feature (Validation)**: Chạy `validate_features.py` để rà soát, loại bỏ các features bị nhiễu, lỗi (NaN, constant), hoặc rò rỉ dữ liệu (data leakage) với LFs.
5. **Huấn luyện mô hình (Training)**: Chạy `train_xgboost_new.py` để tìm tham số tốt nhất với Optuna và lưu lại mô hình XGBoost, sau đó xuất ra các báo cáo đánh giá lỗi (RMSE, MAE, R2).

## Cài đặt và sử dụng (Setup & Usage)

**Yêu cầu hệ thống:** Python 3.8+

**Cài đặt thư viện cần thiết:**
Sử dụng file `requirements.txt` để cài đặt tất cả các thư viện cần thiết:
```bash
pip install -r requirements.txt
```


**Cách chạy pipeline:**
- Bạn có thể chạy từng file Python độc lập qua terminal theo đúng trình tự (từ `make_lf_input.py` đến `train_xgboost_new.py`).
- **Hoặc:** Mở file `fomo_pipeline_eda.ipynb` trong Jupyter Notebook / VS Code, và chạy lần lượt từ trên xuống dưới (file này đã được đồng bộ toàn bộ pipeline).
