import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model

# Tải dữ liệu từ Yahoo Finance
ticker = 'AAPL'  # Bạn có thể thay bằng mã cổ phiếu khác
df = yf.download(ticker, start="2012-01-01", end="2023-01-01")
df = df[['Close']]  # Lấy cột Close
df.index = pd.to_datetime(df.index)  # Chuyển đổi cột Date thành định dạng datetime

# Tiền xử lý dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Chia dữ liệu thành tập train và test
train_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

# Tạo tập dữ liệu cho mô hình
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Reshape dữ liệu
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test))

# Lưu mô hình
model.save('stock_price_prediction_model.h5')

# Đánh giá mô hình
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform([y_test])

# Vẽ biểu đồ giá thực tế và giá dự đoán
actual_length = min(len(df.index[train_data_len+time_step:]), len(y_test_scaled[0]), len(predictions))

plt.figure(figsize=(16,8))
plt.plot(df.index[train_data_len+time_step:train_data_len+time_step+actual_length], y_test_scaled[0][:actual_length], label='Giá thực tế')
plt.plot(df.index[train_data_len+time_step:train_data_len+time_step+actual_length], predictions[:actual_length], label='Giá dự đoán')
plt.title('So sánh giữa giá thực tế và giá dự đoán')
plt.xlabel('Ngày')
plt.ylabel('Giá cổ phiếu')
plt.legend()
plt.savefig('prediction_vs_real_graph.png')  # Lưu biểu đồ dự đoán
plt.show()

# Biểu đồ loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Biểu đồ Loss trong quá trình huấn luyện')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_graph.png')  # Lưu biểu đồ loss
plt.show()

# Tính RMSE và MAE cho tập kiểm tra
rmse = np.sqrt(mean_squared_error(y_test_scaled[0][:actual_length], predictions[:actual_length]))
mae = mean_absolute_error(y_test_scaled[0][:actual_length], predictions[:actual_length])

print('RMSE:', rmse)
print('MAE:', mae)

# Vẽ biểu đồ RMSE và MAE
plt.figure(figsize=(10,6))
plt.bar(['RMSE', 'MAE'], [rmse, mae], color=['blue', 'orange'])
plt.title('Biểu đồ RMSE và MAE')
plt.ylabel('Giá trị')
plt.savefig('rmse_mae_graph.png')  # Lưu biểu đồ RMSE và MAE
plt.show()

# Vẽ biểu đồ hồi quy giữa giá dự đoán và giá thực tế
plt.figure(figsize=(10,6))
sns.regplot(x=y_test_scaled[0][:actual_length], y=predictions[:actual_length], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title('Biểu đồ hồi quy giữa giá dự đoán và giá thực tế')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.savefig('regression_graph.png')  # Lưu biểu đồ hồi quy
plt.show()
