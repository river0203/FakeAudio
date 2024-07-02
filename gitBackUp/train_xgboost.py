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
y_pred = xgb_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'XGBoost Model - Test Accuracy: {accuracy:.4f}')
joblib.dump(xgb_model, 'xgb_model.pkl')
