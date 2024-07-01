import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# 경로 설정 테스트 데이터
test_csv = r"C:\Users\tjdwn\Downloads\open\test_features.csv"
xgb_model_path = r"C:\Users\tjdwn\PycharmProjects\XGBoost\xgb_model1.json"
submission_csv = r"C:\Users\tjdwn\Downloads\open\XGB_submission.csv"

# 테스트 데이터 로드 및 전처리
test_df = pd.read_csv(test_csv)
X_test = test_df.drop(['filename'], axis=1).values

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# XGBoost 모델 로드
xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)

# 예측
predictions = xgb_model.predict_proba(X_test_scaled)

# 결과 저장
results_df = pd.DataFrame()
results_df['id'] = test_df['filename'].apply(lambda x: f'{x.split(".")[0].zfill(5)}')
results_df['fake'] = predictions[:, 1]
results_df['real'] = predictions[:, 0]

# CSV 파일로 저장 (UTF-8 인코딩 적용)
results_df.to_csv(submission_csv, index=False, encoding='utf-8')

print(f"Test results saved to {submission_csv}")