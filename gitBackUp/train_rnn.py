import pandas as pd
import tensorflow as tf

# 데이터 로드
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# RNN 모델 학습 (전체 MFCC와 LFCC 값 사용)
mfcc_lfcc_indices = list(range(52))  # 13 (MFCC mean) + 13 (LFCC mean) + 26 (flattened MFCC and LFCC)

X_train_rnn = X_train[:, mfcc_lfcc_indices]
X_test_rnn = X_test[:, mfcc_lfcc_indices]

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((X_train_rnn.shape[1], 1), input_shape=(X_train_rnn.shape[1],)),
    tf.keras.layers.SimpleRNN(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 추가
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 모델 평가 및 저장
loss, accuracy = model.evaluate(X_test_rnn, y_test)
print(f'RNN Accuracy: {accuracy}')
model.save('rnn_model.h5')
