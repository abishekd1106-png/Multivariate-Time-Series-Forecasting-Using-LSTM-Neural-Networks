import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


df = pd.read_csv("multivariate_timeseries_dataset.csv")
df = df.sort_values("timestamp")

features = ["temp", "humidity", "pressure"]
target = "target"

data = df[features + [target]].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# -------------------------
# 3. Create sequences
# -------------------------
def create_sequences(data, lookback=24):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])   # features only
        y.append(data[i+lookback, -1])      # target
    return np.array(X), np.array(y)

lookback = 24
X, y = create_sequences(data_scaled, lookback)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential()
model.add(LSTM(32, input_shape=(lookback, len(features))))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")


model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)


y_pred = model.predict(X_test)

def inverse_target(y_scaled):
    dummy = np.zeros((len(y_scaled), data.shape[1]))
    dummy[:, -1] = y_scaled
    return scaler.inverse_transform(dummy)[:, -1]

y_test_inv = inverse_target(y_test)
y_pred_inv = inverse_target(y_pred.flatten())


mae = mean_absolute_error(y_test_inv, y_pred_inv)
print("Test MAE:", mae)


plt.figure(figsize=(10,4))
plt.plot(y_test_inv[:200], label="Actual")
plt.plot(y_pred_inv[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted (Simple LSTM)")
plt.show()
