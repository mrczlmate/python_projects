import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna

# -------------------------------
def add_indicators(df):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df

# -------------------------------
def build_attention_lstm_model(input_shape, units, dropout, learning_rate):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(units, return_sequences=True)(inputs)
    lstm_out = Dropout(dropout)(lstm_out)
    attention = Attention()([lstm_out, lstm_out])
    concat = Concatenate()([lstm_out, attention])
    lstm_out_2 = LSTM(units)(concat)
    dropout_2 = Dropout(dropout)(lstm_out_2)
    output = Dense(1)(dropout_2)
    model = Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# -------------------------------
def prepare_data(df, prediction_days):
    features = df[['Close', 'MA50', 'MA200', 'Volume', 'MACD', 'RSI']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    x_data, y_data = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_data.append(scaled_data[x - prediction_days:x, :])
        y_data.append(scaled_data[x, 0])
    return np.array(x_data), np.array(y_data), scaler

# -------------------------------
def objective(trial):
    # Hyperparams
    units = trial.suggest_int('units', 64, 256)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    prediction_days = trial.suggest_int('prediction_days', 50, 200)

    # Data preparation
    data = yf.download('^GSPC', '2012-01-01', '2020-01-01')
    data = add_indicators(data)
    x_train, y_train, scaler = prepare_data(data, prediction_days)

    # Model
    model = build_attention_lstm_model((x_train.shape[1], x_train.shape[2]), units, dropout, learning_rate)
    
    # Train
    history = model.fit(x_train, y_train, epochs=25, batch_size=batch_size, verbose=0)

    # Evaluate on train set itself (mivel csak train van, de ez elegendő az összehasonlításhoz)
    pred = model.predict(x_train)
    pred_prices = scaler.inverse_transform(np.concatenate((pred, np.zeros((pred.shape[0], x_train.shape[2] - 1))), axis=1))[:, 0]
    real_prices = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], x_train.shape[2] - 1))), axis=1))[:, 0]

    rmse = np.sqrt(mean_squared_error(real_prices, pred_prices))
    return rmse

# -------------------------------
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\nBest hyperparameters:")
print(study.best_params)
print(f"Best RMSE: {study.best_value}")
