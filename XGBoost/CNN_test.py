import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# 경로 설정 테스트데이터
test_csv = r"C:\Users\tjdwn\Downloads\open\test_features_filtered.csv"
cnn_model_path = 'cnn_model.h5'
submission_csv = r"C:\Users\tjdwn\Downloads\open\submission4.csv"

# 테스트 데이터 로드 및 전처리
test_df = pd.read_csv(test_csv)
X_test = test_df.drop(['filename'], axis=1).values

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)  # CNN 입력을 위해 차원 확장

# CNN 모델 로드 및 예측
model = load_model(cnn_model_path)
predictions = model.predict(X_test_scaled)

# 예측 결과를 기반으로 제출 파일 생성
submission = pd.DataFrame()
submission['id'] = test_df['filename'].apply(lambda x: f'{x.split(".")[0].zfill(5)}')
submission['fake'] = predictions[:, 0].astype(float)  # 가짜 확률
submission['real'] = predictions[:, 1].astype(float)  # 진짜 확률

# 소수점 형식 지정
submission['fake'] = submission['fake'].apply(lambda x: '{:.4f}'.format(x))
submission['real'] = submission['real'].apply(lambda x: '{:.4f}'.format(x))

# 제출 파일 저장
submission.to_csv(submission_csv, index=False)

print(f'Submission file saved to {submission_csv}')