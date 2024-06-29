import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import joblib
import torch
from RNN import RNNModel

# 파일 경로
test_data_path_with_labels = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/testDataCsv/testDataFeatureExtraction1234.csv'
model_load_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/models/'

# 테스트 데이터 로드
test_data_with_labels = pd.read_csv(test_data_path_with_labels)
# 특징과 라벨 분리 (테스트 데이터)
X_test = test_data_with_labels.drop(['filename', 'voice_type'], axis=1)
y_test = test_data_with_labels['voice_type'].apply(lambda x: 1 if x == 'AI' else 0)

# 모델 불러오기
xgb_model = joblib.load(os.path.join(model_load_path, 'xgb_model.pkl'))
lgb_model = joblib.load(os.path.join(model_load_path, 'lgb_model.pkl'))
scaler_rnn = joblib.load(os.path.join(model_load_path, 'scaler_rnn.pkl'))

# RNN 모델 정의 및 불러오기
input_dim = X_test.shape[1]
hidden_dim = 256
output_dim = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_model = RNNModel(input_dim, hidden_dim, output_dim).to(device)
rnn_model.load_state_dict(torch.load(os.path.join(model_load_path, 'rnn_model.h5')))
rnn_model.eval()

# RNN 모델 예측 준비
X_test_scaled = scaler_rnn.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)

# RNN 예측 수행
with torch.no_grad():
    rnn_outputs = rnn_model(X_test_tensor)
    rnn_pred = (torch.sigmoid(rnn_outputs).cpu().numpy() > 0.5).astype(int).squeeze()

# 다른 모델의 예측 수행
xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)

# 각 모델의 성능 평가
xgb_accuracy = accuracy_score(y_test, xgb_pred)
lgb_accuracy = accuracy_score(y_test, lgb_pred)
rnn_accuracy = accuracy_score(y_test, rnn_pred)

print(f"XGBoost Accuracy: {xgb_accuracy}")
print(f"LightGBM Accuracy: {lgb_accuracy}")
print(f"RNN Accuracy: {rnn_accuracy}")

# 다수결 방식으로 최종 예측 결정
final_pred = []
for xgb, lgb, rnn in zip(xgb_pred, lgb_pred, rnn_pred):
    votes = [xgb, lgb, rnn]
    final_pred.append(1 if votes.count(1) > votes.count(0) else 0)

final_pred = np.array(final_pred)

# 예측 결과를 AI와 REAL로 변환
final_pred_labels = ['AI' if pred == 1 else 'REAL' for pred in final_pred]

# 최종 결과 CSV 파일 생성
result_df = pd.DataFrame({
    'filename': test_data_with_labels['filename'],
    'true_label': test_data_with_labels['voice_type'],
    'xgb_pred': ['AI' if pred == 1 else 'REAL' for pred in xgb_pred],
    'lgb_pred': ['AI' if pred == 1 else 'REAL' for pred in lgb_pred],
    'rnn_pred': ['AI' if pred == 1 else 'REAL' for pred in rnn_pred],
    'final_pred': final_pred_labels
})

result_df_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/testDataResult/testDataFeatureExtraction1234_final.csv'
os.makedirs(os.path.dirname(result_df_path), exist_ok=True)
result_df.to_csv(result_df_path, index=False)

# 평가 결과 출력
print(classification_report(y_test, final_pred, target_names=['REAL', 'AI'], zero_division=0))
print('Final Accuracy with Voting Ensemble:', accuracy_score(y_test, final_pred))
