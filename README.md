# Dự Đoán Giá Cổ Phiếu Sử Dụng Học Sâu (Deep Learning)

## Giới Thiệu
Dự án này xây dựng một mô hình dự đoán giá cổ phiếu sử dụng học sâu, cụ thể là mạng nơ-ron hồi tiếp (LSTM). Mô hình sẽ dự đoán giá cổ phiếu của Apple Inc. (AAPL) dựa trên dữ liệu lịch sử.

## Nội Dung
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Kết Quả](#kết-quả)
- [Giới Thiệu Mô Hình](#giới-thiệu-mô-hình)
- [Tác Giả](#tác-giả)

## Cài Đặt
Để chạy dự án này, bạn cần cài đặt các thư viện sau:

```bash
pip install numpy pandas matplotlib seaborn yfinance tensorflow scikit-learn



Dưới đây là toàn bộ nội dung của file README.md cho dự án dự đoán giá cổ phiếu:

markdown
Copy code
# Dự Đoán Giá Cổ Phiếu Sử Dụng Học Sâu (Deep Learning)

## Giới Thiệu
Dự án này xây dựng một mô hình dự đoán giá cổ phiếu sử dụng học sâu, cụ thể là mạng nơ-ron hồi tiếp (LSTM). Mô hình sẽ dự đoán giá cổ phiếu của Apple Inc. (AAPL) dựa trên dữ liệu lịch sử.

## Nội Dung
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Kết Quả](#kết-quả)
- [Giới Thiệu Mô Hình](#giới-thiệu-mô-hình)
- [Tác Giả](#tác-giả)

## Cài Đặt
Để chạy dự án này, bạn cần cài đặt các thư viện sau:

```bash
pip install numpy pandas matplotlib seaborn yfinance tensorflow scikit-learn


USAGE

- git clone https://github.com/username/stock-price-prediction.git

- cd stock-price-prediction


- python train.py

stock-price-prediction/
│
├── train_model.py               # Code chính để huấn luyện mô hình
├── stock_price_prediction_model.h5 # Mô hình đã huấn luyện
├── prediction_vs_real_graph.png  # Biểu đồ so sánh giữa giá thực tế và giá dự đoán
├── loss_graph.png                # Biểu đồ Loss trong quá trình huấn luyện
├── rmse_mae_graph.png            # Biểu đồ RMSE và MAE
└── regression_graph.png           # Biểu đồ hồi quy giữa giá dự đoán và giá thực tế


-- Kết Quả

*Dưới đây là các biểu đồ kết quả từ mô hình dự đoán:*

- So Sánh giữa Giá Thực Tế và Giá Dự Đoán

- Biểu Đồ Loss

- Biểu Đồ RMSE và MAE

- Biểu Đồ Hồi Quy

-- Giới Thiệu Mô Hình

*Mô hình sử dụng mạng LSTM với:*

- 2 lớp LSTM

- 1 lớp Dense đầu ra

- Mô hình được huấn luyện với dữ liệu lịch sử giá cổ phiếu và sử dụng thuật toán Adam để tối ưu hóa.

**Tác Giả**

Dự án này được thực hiện bởi Nguyễn Huỳnh Hoàng Kha - nhhkha.91tn@gmail.com
