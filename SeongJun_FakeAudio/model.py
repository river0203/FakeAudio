import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# 기존 학습 데이터셋 로드 및 준비
file_path = '/Users/iseongjun/Downloads/resultFile_sorted.csv'  # 학습 데이터셋 경로
data = pd.read_csv(file_path)

# 특징 및 라벨 준비
features = ['average_pitch', 'pitch_variance', 'duration', 'noise_level'] + \
           [f'mfcc_mean_{i}' for i in range(1, 14)] + [f'mfcc_var_{i}' for i in range(1, 14)] + \
           [f'ifcc_mean_{i}' for i in range(1, 14)] + [f'ifcc_var_{i}' for i in range(1, 14)]

X = data[features]
y = data['voice_type'].apply(lambda x: 1 if x == 'AI' else 0).values

# 데이터셋을 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler and feature names
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'feature_names.pkl')

# 로지스틱 회귀 모델 초기화 및 학습
model = LogisticRegression(random_state=42, max_iter=500)
model.fit(X_train_scaled, y_train)

# 모델 평가
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

# 테스트 데이터 경로
test_data_path = '/Users/iseongjun/Downloads/Audio/testData/test_wav'
test_files = [os.path.join(test_data_path, f) for f in os.listdir(test_data_path) if f.endswith('.wav')]

# 특징 추출 함수
def extract_features(audio, sr):
    features = {}

    # 평균 피치 계산
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) == 0:
        pitches = [0]
    features['average_pitch'] = np.mean(pitches)

    # 피치 분산 계산
    features['pitch_variance'] = np.var(pitches)

    # 길이 계산
    features['duration'] = librosa.get_duration(y=audio, sr=sr)

    # 소음 수준 계산 (루트 평균 제곱 에너지로 근사)
    features['noise_level'] = np.mean(librosa.feature.rms(y=audio))

    # MFCC 계산
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(1, 14):
        features[f'mfcc_mean_{i}'] = np.mean(mfccs[i - 1])
        features[f'mfcc_var_{i}'] = np.var(mfccs[i - 1])

    # Inverted Frequency Cepstral Coefficients (IFCCs) 계산 - MFCC를 역변환하여 근사
    ifccs = np.fft.ifft(mfccs, axis=0).real
    for i in range(1, 14):
        features[f'ifcc_mean_{i}'] = np.mean(ifccs[i - 1])
        features[f'ifcc_var_{i}'] = np.var(ifccs[i - 1])

    return features

# Load the scaler and feature names
scaler = joblib.load('scaler.pkl')
saved_feature_names = joblib.load('feature_names.pkl')

# 각 파일에 대해 특징을 추출하고 예측 수행
results = []
for file in test_files:
    audio, sr = librosa.load(file, sr=None)
    features = extract_features(audio, sr)
    features_df = pd.DataFrame([features])
    features_df = features_df[saved_feature_names]  # Ensure the same order of features
    features_scaled = scaler.transform(features_df)

    prediction = model.predict(features_scaled)
    prediction_prob = model.predict_proba(features_scaled)

    results.append({
        'Filename': file,
        'Predicted Label': 'AI' if prediction[0] == 1 else 'Real',
        'Prediction Probability': prediction_prob[0]
    })

# 예측 결과 출력
prediction_results = pd.DataFrame(results)
print(prediction_results)

# 결과를 CSV 파일로 저장
csv_output_path = '/Users/iseongjun/Downloads/Audio/testData/prediction_results.csv'
prediction_results.to_csv(csv_output_path, index=False)

# 결과를 이미지 표로 저장
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=prediction_results.values, colLabels=prediction_results.columns, cellLoc='center', loc='center')

image_output_path = '/Users/iseongjun/Downloads/Audio/testData/prediction_results.png'
plt.savefig(image_output_path)
plt.show()