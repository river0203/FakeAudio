import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import joblib
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RNN import train_and_predict_rnn  # Assuming this function handles hyperparameter tuning internally

# loky 대신 threading 사용
import joblib.parallel
joblib.parallel.parallel_backend('threading')

# 파일 경로
train_data_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv'

# 학습 데이터 로드
print("Loading training data...")
train_data = pd.read_csv(train_data_path)
X_train_full = train_data.drop(['filename', 'voice_type'], axis=1)
y_train_full = train_data['voice_type'].apply(lambda x: 1 if x == 'AI' else 0)

# 데이터 샘플링 (10%)
print("Sampling data (10%)...")
X_train_sample, _, y_train_sample, _ = train_test_split(X_train_full, y_train_full, test_size=0.9, random_state=42, stratify=y_train_full)

# 하이퍼파라미터 튜닝
print("Tuning hyperparameters for SVM...")
svm_param_grid = {'C': [0.1, 1], 'kernel': ['linear']}  # 범위 축소
svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)

svm_grid.fit(X_train_sample, y_train_sample)
best_svm_params = svm_grid.best_params_

print("Tuning hyperparameters for Gradient Boosting...")
gb_param_grid = {'n_estimators': [100], 'learning_rate': [0.01]}  # 범위 축소
gb_grid = GridSearchCV(GradientBoostingClassifier(), gb_param_grid, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)

gb_grid.fit(X_train_sample, y_train_sample)
best_gb_params = gb_grid.best_params_

# 최적 하이퍼파라미터로 전체 데이터 학습
print("Training SVM model with optimal parameters...")
svm_model = SVC(**best_svm_params)

with tqdm(total=1, desc="SVM Training") as pbar:
    svm_model.fit(X_train_full, y_train_full)
    pbar.update(1)

print("Training Gradient Boosting model with optimal parameters...")
gb_model = GradientBoostingClassifier(**best_gb_params)

with tqdm(total=1, desc="GB Training") as pbar:
    gb_model.fit(X_train_full, y_train_full)
    pbar.update(1)

print("Training RNN model with optimal parameters...")
for _ in tqdm(range(20), desc="RNN Training"):
    rnn_model = train_and_predict_rnn(X_train_full, y_train_full, num_epochs=20, learning_rate=0.001)

# 모델 저장
model_save_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/models/'
os.makedirs(model_save_path, exist_ok=True)
joblib.dump(svm_model, os.path.join(model_save_path, 'svm_model.pkl'))
joblib.dump(gb_model, os.path.join(model_save_path, 'gb_model.pkl'))
joblib.dump(rnn_model, os.path.join(model_save_path, 'rnn_model.pkl'))

print("모델 학습이 완료되었고, 모델이 저장되었습니다.")
