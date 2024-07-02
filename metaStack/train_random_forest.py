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

# 학습 데이터에 대한 예측 결과
y_pred_train = rf_model.predict(X_train)
accuracy_train = np.mean(y_pred_train == y_train)
print(f'Random Forest Model - Train Accuracy: {accuracy_train:.4f}')

# 테스트 데이터에 대한 예측 결과
y_pred_test = rf_model.predict(X_test)
accuracy_test = np.mean(y_pred_test == y_test)
print(f'Random Forest Model - Test Accuracy: {accuracy_test:.4f}')

# 모델 저장
joblib.dump(rf_model, 'rf_model.pkl')

# 예측 결과 저장
pd.DataFrame(y_pred_train, columns=['RandomForest']).to_csv('rf_predictions_train.csv', index=False)
pd.DataFrame(y_pred_test, columns=['RandomForest']).to_csv('rf_predictions_test.csv', index=False)
