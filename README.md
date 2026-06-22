# Heartbeat-Classification-IOT-

## Mô tả dự án

Dự án này xây dựng một hệ thống phân loại nhịp tim từ dữ liệu ECG (Electrocardiogram) dựa trên bộ dữ liệu MIT-BIH Arrhythmia. Mục tiêu là phát hiện và phân loại từng nhịp tim thành 5 nhóm nhịp chính bằng mô hình PyTorch.

## Tính năng chính

- Thu thập và tiền xử lý dữ liệu ECG từ MIT-BIH Arrhythmia.
- Trích xuất đoạn nhịp tim quanh các đỉnh QRS.
- Chuẩn hóa và giảm mẫu tín hiệu ECG trước khi đưa vào mô hình.
- Xây dựng tập dữ liệu PyTorch với `Dataset`, `DataLoader` và `WeightedRandomSampler` để xử lý dữ liệu mất cân bằng.
- Huấn luyện mô hình `ECGResNet` với các khối ResBlock và global pooling.
- Lưu mô hình tốt nhất và thực hiện suy luận (inference) trên đoạn ECG mới.

## Cấu trúc dự án

- `src/api.py`: FastAPI application và model inference.
- `notebooks/ECG_Classification.ipynb`: Notebook chính chứa toàn bộ pipeline từ tiền xử lý, tạo dataset, huấn luyện mô hình, đến đánh giá và suy luận.
- `checkpoints/`: Nơi đặt các model checkpoint đã huấn luyện, ví dụ `best_checkpoint.pth`.
- `data/`: Nơi lưu dữ liệu ECG, dataset trung gian hoặc file đầu vào.
- `README.md`: Tài liệu hướng dẫn dự án.

## Dữ liệu

Dự án sử dụng bộ dữ liệu MIT-BIH Arrhythmia từ PhysioNet.

Dataset gồm các file `dat`, `hea`, `atr` được tải về từ:

- https://physionet.org/content/mitdb/

## Nhóm nhãn

Dự án gom các ký hiệu nhịp ECG về 5 nhóm:

- `0`: Normal (N, L, R, e, j)
- `1`: Supraventricular (A, a, J, S)
- `2`: Ventricular (V, E)
- `3`: Paced (F)
- `4`: Other (các nhãn khác)

## Yêu cầu

- Python 3.8+
- PyTorch
- NumPy
- pandas
- SciPy
- matplotlib
- scikit-learn
- wfdb
- tqdm
- torchinfo
- fastapi
- uvicorn

## Cài đặt nhanh

```bash
pip install torch numpy pandas scipy matplotlib scikit-learn wfdb tqdm torchinfo fastapi uvicorn
```

## FastAPI inference

1. Đặt checkpoint `best_checkpoint` vào thư mục `checkpoints/`. File có thể có tên `best_checkpoint` hoặc `best_checkpoint.pth`.
2. Chạy API:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

3. Gửi yêu cầu POST đến `http://127.0.0.1:8000/predict` với body JSON:

```json
{
  "ecg_signal": [0.1, 0.05, -0.02, ...],
  "sampling_rate": 360
}
```

4. Kết quả trả về gồm danh sách dự đoán từng nhịp, dự đoán chung theo majority vote và xác suất trung bình mỗi lớp.

## Hướng dẫn sử dụng

1. Mở `ECG_Classification.ipynb` trong Jupyter Notebook / Colab.
2. Cài đặt các thư viện cần thiết.
3. Tải bộ dữ liệu MIT-BIH Arrhythmia về thư mục tương ứng hoặc cấu hình đúng đường dẫn `record_path` trong notebook.
4. Chạy các cell theo thứ tự:
   - Import thư viện.
   - Tiền xử lý và tạo dataset.
   - Tạo PyTorch Dataset & DataLoader.
   - Định nghĩa mô hình `ECGResNet`.
   - Huấn luyện và đánh giá mô hình.
   - Lưu mô hình và chạy inference.

## Luồng xử lý chính

1. Đọc ECG và annotation bằng `wfdb`.
2. Xác định các đỉnh nhịp (`peaks`), trích xuất đoạn tín hiệu quanh mỗi nhịp.
3. Chuẩn hóa, giảm mẫu và gắn nhãn cho mỗi beat.
4. Chia dữ liệu train/validation/test theo tỷ lệ 70/15/15.
5. Sử dụng `WeightedRandomSampler` để cân bằng mẫu trong huấn luyện.
6. Huấn luyện `ECGResNet` với hàm mất mát `CrossEntropyLoss` và tối ưu Adam.
7. Đánh giá kết quả bằng `accuracy`, `F1-score` và báo cáo phân loại.

## Mô hình

Mô hình `ECGResNet` gồm:

- 3 khối residual 1D convolutional (`ResBlock`).
- Global max pooling và global average pooling.
- Fully connected layer + softmax.

## Inference

- Lưu mô hình sau huấn luyện với `torch.save(model.state_dict(), "resetECG.pth")`.
- Tải mô hình và áp dụng trên đoạn ECG mới.
- Tách đoạn ECG thành các beat, đưa vào mô hình để dự đoán từng beat.
- Tổng hợp dự đoán bằng majority vote hoặc xác suất trung bình.

## Ghi chú

- Notebook hiện tại được viết cho môi trường `Colab` / Jupyter với đường dẫn `/content`.
- Nếu chạy cục bộ, hãy điều chỉnh đường dẫn đầu vào dữ liệu và file kết quả.
- Dữ liệu ECG thô có thể cần thêm bước phát hiện peak và lọc nhiễu tùy theo nguồn dữ liệu.

## Liên hệ

Nếu cần mở rộng dự án, bạn có thể bổ sung:

- Thêm dữ liệu ECG khác.
- Tăng cường dữ liệu (augmentation) cho tập nhịp.
- Cải tiến kiến trúc mạng hoặc thêm attention.
- Xây dựng API REST để chạy mô hình online.
