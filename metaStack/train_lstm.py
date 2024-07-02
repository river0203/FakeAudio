import numpy as np
import pandas as pd
import tensorflow as tf

# 데이터 로드
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# LSTM 모델 학습
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # 학습률을 적절히 조정

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 추가
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# 모델 평가 및 저장
loss, accuracy = model.evaluate(X_test, y_test)
print(f'LSTM Accuracy: {accuracy}')
model.save('lstm_model.h5')

# LSTM 모델 예측 결과 저장
X_train_lstm = np.expand_dims(X_train, axis=2)
X_test_lstm = np.expand_dims(X_test, axis=2)
y_pred_train_lstm = model.predict(X_train_lstm)
y_pred_test_lstm = model.predict(X_test_lstm)
y_pred_train_lstm = (y_pred_train_lstm > 0.5).astype(int)
y_pred_test_lstm = (y_pred_test_lstm > 0.5).astype(int)

# 예측 결과를 CSV 파일로 저장
pd.DataFrame(y_pred_train_lstm, columns=['LSTM']).to_csv('lstm_predictions_train.csv', index=False)
pd.DataFrame(y_pred_test_lstm, columns=['LSTM']).to_csv('lstm_predictions_test.csv', index=False)
