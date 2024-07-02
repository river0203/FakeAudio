import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import joblib

# 데이터 로드
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# XGBoost 모델 학습
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# 모델 평가 및 저장
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# 정확도 계산
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)
print(f'XGBoost Model - Train Accuracy: {accuracy_train:.4f}')
print(f'XGBoost Model - Test Accuracy: {accuracy_test:.4f}')

# 모델 저장
joblib.dump(xgb_model, 'xgb_model.pkl')

# 예측 결과 저장
pd.DataFrame(y_pred_train, columns=['XGBoost']).to_csv('xgb_predictions_train.csv', index=False)
pd.DataFrame(y_pred_test, columns=['XGBoost']).to_csv('xgb_predictions_test.csv', index=False)
