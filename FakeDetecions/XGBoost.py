import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 데이터 로드
data = pd.read_csv('C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv')

# 특징과 레이블 분리
X = data.drop(columns=['filename', 'voice_type'])
y = data['voice_type'].apply(lambda x: 1 if x == 'AI' else 0)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(f'XGBoost Accuracy: {accuracy_score(y_test, y_pred)}')
