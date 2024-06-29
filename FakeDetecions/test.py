import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['filename', 'voice_type'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, data['filename']

# 모델 로드 및 예측 수행
def load_and_predict(model, X_test, model_type):
    if model_type in ['rnn', 'lstm', 'cnn']:
        X_test = np.expand_dims(X_test, axis=2)
    return (model.predict(X_test) > 0.5).astype(int).reshape(-1) if model_type in ['rnn', 'lstm', 'cnn'] else model.predict(X_test)

# 모델 학습
def train_models(X_train, y_train):
    models = {}

    # XGBoost
    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(X_train, y_train)
    models['xgb'] = model_xgb

    # Random Forests
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    models['rf'] = model_rf

    # RNN
    model_rnn = tf.keras.Sequential([
        tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
        tf.keras.layers.SimpleRNN(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_rnn.fit(X_train, y_train, epochs=10, batch_size=32)
    models['rnn'] = model_rnn

    # LSTM
    model_lstm = tf.keras.Sequential([
        tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm.fit(X_train, y_train, epochs=10, batch_size=32)
    models['lstm'] = model_lstm

    # CNN
    input_shape = (X_train.shape[1], 1)
    X_train_cnn = np.expand_dims(X_train, axis=2)
    model_cnn = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32)
    models['cnn'] = model_cnn

    return models

# 앙상블 예측 수행
def ensemble_predict(models, X_test):
    predictions = np.zeros((X_test.shape[0], len(models)))

    for i, (model_name, model) in enumerate(models.items()):
        pred = load_and_predict(model, X_test, model_name)
        predictions[:, i] = pred

    final_pred = (predictions.sum(axis=1) > (len(models) / 2)).astype(int)
    return final_pred

# 결과 저장
def save_results(filenames, predictions, output_file):
    results = pd.DataFrame({'filename': filenames, 'prediction': predictions})
    results.to_csv(output_file, index=False)

def main():
    # 테스트 데이터 로드 및 전처리
    test_data_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/testDataCsv/123.csv'
    X_test, filenames = load_and_preprocess_data(test_data_path)

    # 학습 데이터 로드 및 전처리 (훈련을 위해 필요)
    train_data_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv'
    train_data = pd.read_csv(train_data_path)
    X_train = train_data.drop(columns=['filename', 'voice_type'])
    y_train = train_data['voice_type'].apply(lambda x: 1 if x == 'AI' else 0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 모델 학습
    models = train_models(X_train_scaled, y_train)

    # 앙상블 예측 수행
    final_pred = ensemble_predict(models, X_test)

    # 결과 저장
    output_file = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/testDataCsv/predictions.csv'
    save_results(filenames, final_pred, output_file)

if __name__ == "__main__":
    main()
