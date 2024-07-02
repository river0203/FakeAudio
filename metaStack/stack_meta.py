import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# 각 모델의 학습 예측 결과 로드
y_pred_cnn_train = pd.read_csv('cnn_predictions_train.csv').values.flatten()
y_pred_xgb_train = pd.read_csv('xgb_predictions_train.csv').values.flatten()
y_pred_rf_train = pd.read_csv('rf_predictions_train.csv').values.flatten()
y_pred_lstm_train = pd.read_csv('lstm_predictions_train.csv').values.flatten()
y_pred_rnn_train = pd.read_csv('rnn_predictions_train.csv').values.flatten()
y_pred_gmm_train = pd.read_csv('gmm_predictions_train.csv').values.flatten()

# 메타 모델 학습 데이터 생성
X_meta_train = pd.DataFrame({
    'CNN': y_pred_cnn_train,
    'XGBoost': y_pred_xgb_train,
    'RandomForest': y_pred_rf_train,
    'LSTM': y_pred_lstm_train,
    'RNN': y_pred_rnn_train,
    'GMM': y_pred_gmm_train
})

# 실제 레이블 로드
y_train = pd.read_csv('y_train.csv').values.flatten()

# 고급 스태킹 모델 생성
estimators = [
    ('cnn', LogisticRegression()),
    ('xgb', LogisticRegression()),
    ('rf', LogisticRegression()),
    ('lstm', LogisticRegression()),
    ('rnn', LogisticRegression()),
    ('gmm', LogisticRegression())
]

# 메타 모델을 RandomForestClassifier로 설정
final_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# 스태킹 모델 생성
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=KFold(n_splits=5)
)

# 메타 모델 학습
stacking_model.fit(X_meta_train, y_train)

# 피처 이름 저장
feature_names = X_meta_train.columns
joblib.dump(feature_names, 'meta_model_features.pkl')

# 메타 모델 저장
joblib.dump(stacking_model, 'meta_model_stack.pkl')
