import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib
import sys
import lightgbm as lgb
import xgboost as xgb
import  torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RNN import train_and_predict_rnn


# 파일 경로
train_data_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv'

# 학습 데이터 로드
print("Loading training data...")
train_data = pd.read_csv(train_data_path)
X_train_full = train_data.drop(['filename', 'voice_type'], axis=1)
y_train_full = train_data['voice_type'].apply(lambda x: 1 if x == 'AI' else 0)

# 데이터 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)

# 하이퍼파라미터 튜닝 및 모델 학습
print("Tuning hyperparameters for XGBoost...")
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}
xgb_grid = GridSearchCV(xgb.XGBClassifier(), xgb_param_grid, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)

xgb_grid.fit(X_train_scaled, y_train_full)
best_xgb_params = xgb_grid.best_params_

print("Tuning hyperparameters for LightGBM...")
lgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'num_leaves': [31, 40, 50],
    'min_child_samples': [20, 30, 40]
}
lgb_grid = GridSearchCV(lgb.LGBMClassifier(), lgb_param_grid, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)

lgb_grid.fit(X_train_scaled, y_train_full)
best_lgb_params = lgb_grid.best_params_

# 최적 하이퍼파라미터로 전체 데이터 학습
print("Training XGBoost model with optimal parameters...")
xgb_model = xgb.XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train_scaled, y_train_full)

print("Training LightGBM model with optimal parameters...")
lgb_model = lgb.LGBMClassifier(**best_lgb_params)
lgb_model.fit(X_train_scaled, y_train_full)

print("Training RNN model with optimal parameters...")
rnn_model, scaler_rnn = train_and_predict_rnn(
    X_train_scaled, y_train_full, X_train_scaled,
    num_epochs=50,  # 증가된 에폭 수
    learning_rate=0.0001,  # 감소된 학습률
    hidden_dim=512,  # 증가된 히든 레이어 크기
    return_model=True
)

# 모델 저장
model_save_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/models/'
os.makedirs(model_save_path, exist_ok=True)
joblib.dump(xgb_model, os.path.join(model_save_path, 'xgb_model.pkl'))
joblib.dump(lgb_model, os.path.join(model_save_path, 'lgb_model.pkl'))
torch.save(rnn_model.state_dict(), os.path.join(model_save_path, 'rnn_model.h5'))
joblib.dump(scaler_rnn, os.path.join(model_save_path, 'scaler_rnn.pkl'))

print("모델 학습이 완료되었고, 모델이 저장되었습니다.")
