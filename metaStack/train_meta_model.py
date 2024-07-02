import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier

# 데이터 로드
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# 각 모델의 예측 결과 로드
y_pred_cnn = pd.read_csv('cnn_predictions.csv').values.flatten()

# XGBoost 모델 예측 결과
xgb_model = joblib.load('xgb_model.pkl')
y_pred_xgb = xgb_model.predict(X_test)

# Random Forest 모델 예측 결과
rf_model = joblib.load('rf_model.pkl')
y_pred_rf = rf_model.predict(X_test)

# 각 모델의 예측 결과를 특징으로 사용하여 메타 모델 학습 데이터 생성
X_meta_train = pd.DataFrame({
    'CNN': y_pred_cnn,
    'XGBoost': y_pred_xgb,
    'RandomForest': y_pred_rf
})

# 고급 스태킹 모델 생성
estimators = [
    ('cnn', LogisticRegression()),
    ('xgb', xgb_model),
    ('rf', rf_model)
]

meta_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=KFold(n_splits=5))

# 메타 모델 학습
meta_model.fit(X_meta_train, y_test)

# 메타 모델 평가
y_meta_pred = meta_model.predict(X_meta_train)
meta_accuracy = accuracy_score(y_test, y_meta_pred)
print(f'Meta Model Accuracy: {meta_accuracy:.4f}')

# 메타 모델 저장
joblib.dump(meta_model, 'meta_model.pkl')
