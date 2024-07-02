import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import joblib

# 경로 설정 테스트 데이터
test_data_folder = r"C:\Users\tjdwn\Downloads\open\test"
sample_submission_csv = r"C:\Users\tjdwn\Downloads\open\sample_submission.csv"

# 오디오 특징 추출 함수 (증강 없이)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    n_fft = min(512, len(y) // 2)  # 오디오 길이에 따라 n_fft 조정
    hop_length = max(256, n_fft // 2)  # hop_length는 n_fft의 절반 또는 최소값 256

    # MFCC 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # LFCC 추출
    lfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    lfccs_mean = np.mean(lfccs.T, axis=0)

    # Pitch 계산
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    pitches = pitches[pitches > 0]
    if len(pitches) == 0:
        avg_pitch = 0
        pitch_var = 0
    else:
        avg_pitch = np.mean(pitches)
        pitch_var = np.var(pitches)

    features = np.hstack([mfccs_mean, lfccs_mean, avg_pitch, pitch_var])
    return features

# 테스트 데이터 로드 및 특징 추출
def create_test_dataset(test_data_folder):
    features = []
    filenames = []

    for filename in os.listdir(test_data_folder):
        if filename.endswith('.ogg'):
            file_path = os.path.join(test_data_folder, filename)
            feature_vector = extract_features(file_path)
            features.append(feature_vector)
            filenames.append(filename)

    X_test = np.array(features)
    return X_test, filenames

# 모델 로드
cnn_model = tf.keras.models.load_model('cnn_model.h5')
xgb_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')
meta_model = joblib.load('meta_model.pkl')

# 테스트 데이터 생성
X_test, filenames = create_test_dataset(test_data_folder)

# 테스트 데이터 저장
test_features_df = pd.DataFrame(X_test, index=filenames)
test_features_df.to_csv('test_features.csv', index=True)

# 필요한 특징 선택
mfcc_mean_indices = list(range(13))  # MFCC mean 값이 첫 13개의 특징

# 모델 예측
X_test_cnn = X_test[:, mfcc_mean_indices]
X_test_cnn = np.expand_dims(X_test_cnn, axis=2)  # CNN 입력 형식에 맞게 데이터 차원 확장
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = (y_pred_cnn > 0.5).astype(int).flatten()

y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# 메타 모델 예측
X_meta_test = pd.DataFrame({
    'CNN': y_pred_cnn,
    'XGBoost': y_pred_xgb,
    'RandomForest': y_pred_rf
})

y_pred_meta = meta_model.predict_proba(X_meta_test)[:, 1]

# 결과 저장
submission = pd.read_csv(sample_submission_csv)
submission['fake'] = y_pred_meta
submission['real'] = 1 - y_pred_meta

submission.to_csv('submission.csv', index=False, encoding='utf-8')
print("Predictions saved to submission.csv")
