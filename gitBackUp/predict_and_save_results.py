import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# 경로 설정
test_features_csv = 'test_features3.csv'
sample_submission_csv = r"C:\Users\tjdwn\Downloads\open\sample_submission.csv"
output_submission_csv = r"C:\Users\tjdwn\Downloads\open\submission4th.csv"

# 모델 로드
cnn_model = tf.keras.models.load_model('cnn_model.h5')
xgb_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')
meta_model = joblib.load('meta_model.pkl')

# 테스트 데이터 로드
X_test = pd.read_csv(test_features_csv, index_col=0).values
filenames = pd.read_csv(test_features_csv, index_col=0).index

# CNN 모델 예측
X_test_cnn = np.expand_dims(X_test, axis=2)  # CNN 입력 형식에 맞게 데이터 차원 확장
y_pred_cnn = cnn_model.predict(X_test_cnn)

# XGBoost 모델 예측
y_pred_xgb = xgb_model.predict(X_test)

# Random Forest 모델 예측
y_pred_rf = rf_model.predict(X_test)

# 메타 모델 예측
X_meta_test = pd.DataFrame({
    'CNN': y_pred_cnn.flatten(),
    'XGBoost': y_pred_xgb,
    'RandomForest': y_pred_rf
})

y_pred_meta = meta_model.predict_proba(X_meta_test)

# 결과 저장을 위한 DataFrame 생성
submission = pd.read_csv(sample_submission_csv)

# 예측된 값을 사용하여 'fake' 및 'real' 열을 채우기
submission['fake'] = y_pred_meta[:, 1]  # 가짜 목소리 확률
submission['real'] = y_pred_meta[:, 0]  # 진짜 목소리 확률

# 확률에 따른 결과 설정
for i in range(len(submission)):
    fake_prob = submission.loc[i, 'fake']
    real_prob = submission.loc[i, 'real']

    if 0.4 <= fake_prob <= 0.6 and 0.4 <= real_prob <= 0.6:
        submission.loc[i, ['fake', 'real']] = [1, 1]
    elif fake_prob > 0.6 and real_prob > 0.6:
        submission.loc[i, ['fake', 'real']] = [1, 1]
    elif fake_prob > 0.6:
        submission.loc[i, ['fake', 'real']] = [1, 0]
    elif real_prob > 0.6:
        submission.loc[i, ['fake', 'real']] = [0, 1]
    else:
        submission.loc[i, ['fake', 'real']] = [0, 0]

# 아무 소리도 없는 경우 처리: 모든 특징 값이 0인 경우
for i, row in enumerate(X_test):
    if np.sum(row) == 0:
        submission.loc[i, 'fake'] = 0
        submission.loc[i, 'real'] = 0

# 결과 저장
submission.to_csv(output_submission_csv, index=False, encoding='utf-8')
print(f"Predictions saved to {output_submission_csv}")
