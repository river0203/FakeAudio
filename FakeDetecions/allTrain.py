import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

# 데이터 로드
data = pd.read_csv('C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv')

# 특징과 레이블 분리
X = data.drop(columns=['filename', 'voice_type'])
y = data['voice_type'].apply(lambda x: 1 if x == 'AI' else 0)

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost 모델 학습
model_xgb = xgb.XGBClassifier()
model_xgb.fit(X_train, y_train)
xgb_pred = model_xgb.predict(X_test)

# Random Forests 모델 학습
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
rf_pred = model_rf.predict(X_test)

# RNN 모델 학습
model_rnn = tf.keras.Sequential([
    tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.SimpleRNN(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32)
rnn_pred = (model_rnn.predict(X_test) > 0.5).astype(int).reshape(-1)

# LSTM 모델 학습
model_lstm = tf.keras.Sequential([
    tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32)
lstm_pred = (model_lstm.predict(X_test) > 0.5).astype(int).reshape(-1)

# CNN 모델 학습 (1D CNN)
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32)
cnn_pred = (model_cnn.predict(X_test_cnn) > 0.5).astype(int).reshape(-1)

# 다수결 투표
final_pred = np.array([1 if np.sum(pred) > 2 else 0 for pred in zip(xgb_pred, rf_pred, rnn_pred, lstm_pred, cnn_pred)])

# 최종 성능 평가
print(f'Ensemble Accuracy: {accuracy_score(y_test, final_pred)}')
