import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 로드
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# CNN 모델 학습
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 모델 평가 및 저장
loss, accuracy = model.evaluate(X_test, y_test)
print(f'CNN Accuracy: {accuracy}')
model.save('cnn_model.h5')

# CNN 모델 예측 결과 저장
X_test_cnn = np.expand_dims(X_test, axis=2)  # CNN 입력 형식에 맞게 데이터 차원 확장
y_pred_cnn = model.predict(X_test_cnn)
y_pred_cnn = (y_pred_cnn > 0.5).astype(int)
pd.DataFrame(y_pred_cnn, columns=['CNN']).to_csv('cnn_predictions.csv', index=False)
