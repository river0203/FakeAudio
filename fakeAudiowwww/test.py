import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 파일 경로
test_data_path_with_labels = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/testDataCsv/testDataFeatureExtraction1234.csv'
model_load_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/models/'

# 테스트 데이터 로드
test_data_with_labels = pd.read_csv(test_data_path_with_labels)
# 특징과 라벨 분리 (테스트 데이터)
X_test = test_data_with_labels.drop(['filename', 'voice_type'], axis=1)
y_test = test_data_with_labels['voice_type'].apply(lambda x: 1 if x == 'AI' else 0)

# 모델 불러오기
svm_model = joblib.load(os.path.join(model_load_path, 'svm_model.pkl'))
gb_model = joblib.load(os.path.join(model_load_path, 'gb_model.pkl'))
rnn_model = joblib.load(os.path.join(model_load_path, 'rnn_model.pkl'))

# 예측 수행
svm_pred = svm_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
rnn_pred = rnn_model.predict(X_test)

# 각 모델의 성능 평가
svm_accuracy = accuracy_score(y_test, svm_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
rnn_accuracy = accuracy_score(y_test, rnn_pred)

print(f"SVM Accuracy: {svm_accuracy}")
print(f"Gradient Boosting Accuracy: {gb_accuracy}")
print(f"RNN Accuracy: {rnn_accuracy}")

# 성능에 비례한 가중치 계산 (간단히 정확도를 사용한 예)
total_accuracy = svm_accuracy + gb_accuracy + rnn_accuracy
svm_weight = svm_accuracy / total_accuracy
gb_weight = gb_accuracy / total_accuracy
rnn_weight = rnn_accuracy / total_accuracy

# 가중치를 적용한 최종 예측
final_pred = np.round((svm_pred * svm_weight + gb_pred * gb_weight + rnn_pred * rnn_weight)).astype(int)

# 예측 결과를 AI와 REAL로 변환
final_pred_labels = ['AI' if pred == 1 else 'REAL' for pred in final_pred]

# 최종 결과 CSV 파일 생성
result_df = pd.DataFrame({
    'filename': test_data_with_labels['filename'],
    'true_label': test_data_with_labels['voice_type'],
    'svm_pred': ['AI' if pred == 1 else 'REAL' for pred in svm_pred],
    'gb_pred': ['AI' if pred == 1 else 'REAL' for pred in gb_pred],
    'rnn_pred': ['AI' if pred == 1 else 'REAL' for pred in rnn_pred],
    'final_pred': final_pred_labels
})

result_df_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/testDataResult/final_prediction_results_weighted.csv'
os.makedirs(os.path.dirname(result_df_path), exist_ok=True)
result_df.to_csv(result_df_path, index=False)

# 평가 결과 출력
print(classification_report(y_test, final_pred, target_names=['REAL', 'AI'], zero_division=0))
print('Final Accuracy with Weighted Ensemble:', accuracy_score(y_test, final_pred))
