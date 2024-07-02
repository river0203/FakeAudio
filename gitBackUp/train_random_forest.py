import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 데이터 로드
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# Random Forest 모델 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 모델 평가 및 저장
y_pred = rf_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Random Forest Model - Test Accuracy: {accuracy:.4f}')
joblib.dump(rf_model, 'rf_model.pkl')
