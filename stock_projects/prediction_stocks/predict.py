import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import tensorflow as tf
import optuna
from finta import TA

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from tensorflow.keras import backend as K

# -------------------------------
# Feature engineering with finta indicators

def add_indicators(df):
    df = df.copy()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df['SMA_50'] = TA.SMA(df, 50)
    df['SMA_200'] = TA.SMA(df, 200)
    df['EMA_12'] = TA.EMA(df, 12)
    df['EMA_26'] = TA.EMA(df, 26)
    df['MACD'] = TA.MACD(df)['MACD']
    df['RSI'] = TA.RSI(df)

    bb = TA.BBANDS(df)
    df['BBU'] = bb['BB_UPPER']
    df['BBL'] = bb['BB_LOWER']

    df['ATR'] = TA.ATR(df)

    columns_required = ['Close', 'SMA_50', 'SMA_200', 'Volume', 'MACD', 'RSI', 'BBU', 'BBL', 'ATR']
    df = df.dropna(subset=columns_required)

    if df.empty:
        raise ValueError("No valid rows remaining after applying indicators.")

    return df

# -------------------------------
# Model building - Optuna version

def build_gru_model(input_shape, gru_units, dense_units, dropout_rate):
    inputs = Input(shape=input_shape)
    gru_out = GRU(gru_units, return_sequences=False)(inputs)
    dropout = Dropout(dropout_rate)(gru_out)
    dense = Dense(dense_units, activation='relu')(dropout)
    output = Dense(1)(dense)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# -------------------------------
# Data download and preprocessing

company = '^GSPC'
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2020, 1, 1)

data = yf.download(company, start, end)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = add_indicators(data)

features_used = ['Close', 'SMA_50', 'SMA_200', 'Volume', 'MACD', 'RSI', 'BBU', 'BBL', 'ATR']
features = data[features_used].values

feature_scaler = MinMaxScaler()
scaled_data = feature_scaler.fit_transform(features)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(data[['Close']].values)

prediction_days = 100
x, y = [], []

for i in range(prediction_days, len(scaled_data)):
    x.append(scaled_data[i - prediction_days:i, :])
    y.append(scaled_close[i, 0])

x, y = np.array(x), np.array(y)

# -------------------------------
# Optuna objective function

def objective(trial):
    gru_units = trial.suggest_int("gru_units", 32, 128)
    dense_units = trial.suggest_int("dense_units", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    split_index = int(len(x) * 0.9)
    x_train, x_val = x[:split_index], x[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    model = build_gru_model((x.shape[1], x.shape[2]), gru_units, dense_units, dropout_rate)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0,
              validation_data=(x_val, y_val), shuffle=False, callbacks=[early_stop])

    val_pred = model.predict(x_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"Trial {trial.number}: RMSE={val_rmse:.6f}, params={{'gru_units': {gru_units}, 'dense_units': {dense_units}, 'dropout_rate': {dropout_rate:.3f}}}")

    K.clear_session()
    return val_rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, show_progress_bar=True)
best_params = study.best_params

# -------------------------------
# Train new model with best hyperparameters

model = build_gru_model((x.shape[1], x.shape[2]),
                        best_params['gru_units'],
                        best_params['dense_units'],
                        best_params['dropout_rate'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

split_index = int(len(x) * 0.9)
x_train, x_val = x[:split_index], x[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0,
          validation_data=(x_val, y_val), shuffle=False, callbacks=[early_stop, TqdmCallback(verbose=1)])

# -------------------------------
# Test data preparation

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(company, test_start, test_end)
if isinstance(test_data.columns, pd.MultiIndex):
    test_data.columns = test_data.columns.get_level_values(0)
test_data = add_indicators(test_data)
actual_prices = test_data['Close'].values

total_data = pd.concat((data, test_data), axis=0)
recent_data = total_data.loc[test_data.index[0] - pd.Timedelta(days=prediction_days):]
model_inputs = recent_data[features_used].values
model_inputs = feature_scaler.transform(model_inputs)

x_test = []
for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i - prediction_days:i, :])
x_test = np.array(x_test)

# -------------------------------
# Prediction and inverse scaling

predicted_scaled = model.predict(x_test)
predicted_prices = close_scaler.inverse_transform(predicted_scaled)[:, 0]

rmse = np.sqrt(mean_squared_error(actual_prices[-len(predicted_prices):], predicted_prices))
print(f"RMSE on test data: {rmse:.2f}")

# -------------------------------
# Visualization

plt.figure(figsize=(14,6))
plt.plot(actual_prices, color='black', label='Actual Price')
plt.plot(predicted_prices, color='green', label='Predicted Price (Optimized GRU)')
plt.title(f'{company} Share Price - Optimized GRU Model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Error distribution

errors = actual_prices[-len(predicted_prices):] - predicted_prices
plt.figure()
plt.hist(errors, bins=50, color='gray')
plt.title("Prediction Error Histogram")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -------------------------------
# MAPE (relative error)

mape = np.mean(np.abs(errors / actual_prices[-len(predicted_prices):])) * 100
print(f"MAPE on test data: {mape:.2f}%")