import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import time

# 시작 시간 기록
start_time = time.time()

print("데이터 로딩 시작...")
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
print("데이터 로딩 완료.")

print("데이터 분할 시작...")
# 훈련, 검증, 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X_meta_train, y_train, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
print("데이터 분할 완료.")

print("데이터 전처리 시작...")
# 스케일링 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("데이터 전처리 완료.")

print("스태킹 모델 정의 시작...")
# 스태킹 모델 생성
estimators = [
    ('cnn', RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ('xgb', RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ('rf', RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ('lstm', RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ('rnn', RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ('gmm', RidgeCV(alphas=[0.1, 1.0, 10.0]))
]

# 메타 모델 설정 (RidgeCV 사용)
final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0])

# 스태킹 모델 생성
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)
print("스태킹 모델 정의 완료.")

print("스태킹 모델 학습 시작...")
# 메타 모델 학습
stacking_model.fit(X_train_scaled, y_train)
print("스태킹 모델 학습 완료.")

print("모델 성능 평가 시작...")
# 모델 성능 평가
y_val_pred = stacking_model.predict(X_val_scaled)
mse_val = mean_squared_error(y_val, y_val_pred)
print(f"검증 데이터에 대한 평균 제곱 오차: {mse_val}")

y_test_pred = stacking_model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"테스트 데이터에 대한 평균 제곱 오차: {mse_test}")
print("모델 성능 평가 완료.")

print("모델 및 관련 정보 저장 시작...")
# 스케일러 저장
joblib.dump(scaler, 'meta_model_scaler.pkl')

# 메타 모델 저장
joblib.dump(stacking_model, 'meta_model_stack.pkl')

# 예측 함수 (확률 값 반환)
def predict_proba(X):
    X_scaled = scaler.transform(X)
    return stacking_model.predict(X_scaled)

# 예측 함수 저장
joblib.dump(predict_proba, 'meta_model_predict_proba.pkl')
print("모델 및 관련 정보 저장 완료.")

# 종료 시간 기록 및 총 실행 시간 계산
end_time = time.time()
total_time = end_time - start_time

print(f"총 실행 시간: {total_time:.2f}초")
print("모든 과정이 완료되었습니다.")