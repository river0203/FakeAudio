import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import joblib
import numpy as np
import pandas as pd

# 경로 설정
test_features_csv = 'test_features3.csv'
sample_submission_csv = r"C:\Users\tjdwn\Downloads\open\sample_submission.csv"
output_submission_csv = r"C:\Users\tjdwn\Downloads\open\submission11th.csv"

# 모델 로드
custom_objects = {"adam": Adam}
cnn_model = tf.keras.models.load_model('cnn_model.h5', custom_objects=custom_objects)
lstm_model = tf.keras.models.load_model('lstm_model.h5', custom_objects=custom_objects)
rnn_model = tf.keras.models.load_model('rnn_model.h5', custom_objects=custom_objects)
xgb_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')
gmm_model = joblib.load('gmm_model.pkl')
meta_model = joblib.load('meta_model_stack.pkl')
scaler = joblib.load('meta_model_scaler.pkl')

# 테스트 데이터 로드
X_test = pd.read_csv(test_features_csv, index_col=0).values
filenames = pd.read_csv(test_features_csv, index_col=0).index

# 파일 확장자 제거
filenames = [filename.split('.')[0] for filename in filenames]

# CNN 모델 예측
X_test_cnn = np.expand_dims(X_test, axis=2)
y_pred_cnn = cnn_model.predict(X_test_cnn).flatten()

# LSTM 모델 예측
X_test_lstm = np.expand_dims(X_test, axis=2)
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

# RNN 모델 예측
X_test_rnn = np.expand_dims(X_test, axis=2)
y_pred_rnn = rnn_model.predict(X_test_rnn).flatten()

# XGBoost 모델 예측
y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Random Forest 모델 예측
y_pred_rf = rf_model.predict_proba(X_test)[:, 1]

# GMM 모델 예측
y_pred_gmm = gmm_model.predict_proba(X_test)[:, 1]

# 메타 모델 예측을 위한 데이터프레임 생성
X_meta_test = pd.DataFrame({
    'CNN': y_pred_cnn,
    'LSTM': y_pred_lstm,
    'RNN': y_pred_rnn,
    'XGBoost': y_pred_xgb,
    'RandomForest': y_pred_rf,
    'GMM': y_pred_gmm
})

# 스케일러의 특성 이름 확인
if hasattr(scaler, 'get_feature_names_out'):
    scaler_features = scaler.get_feature_names_out()
else:
    scaler_features = scaler.feature_names_in_

print("Scaler features:", scaler_features)
print("X_meta_test features:", X_meta_test.columns)

# X_meta_test를 스케일러의 특성에 맞게 조정
X_meta_test = X_meta_test.reindex(columns=scaler_features, fill_value=0)

print("Adjusted X_meta_test features:", X_meta_test.columns)

# 스케일링 적용
X_meta_test_scaled = scaler.transform(X_meta_test)

# 메타 모델로 최종 예측
y_pred_meta = meta_model.predict(X_meta_test_scaled)

# 제출 파일 생성 및 소수점 네 자리로 반올림
submission = pd.DataFrame({
    'id': filenames,
    'fake': np.round(y_pred_meta, 4),
    'real': np.round(1 - y_pred_meta, 4)
})

# 모든 특성이 0인 경우 처리
for i, row in enumerate(X_test):
    if np.sum(row) == 0:
        submission.loc[i, 'fake'] = 0.0000
        submission.loc[i, 'real'] = 0.0000

# 제출 파일 저장
submission.to_csv(output_submission_csv, index=False, encoding='utf-8', float_format='%.4f')
print(f"Predictions saved to {output_submission_csv}")