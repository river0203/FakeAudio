import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 오디오 데이터 로드
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# MFCC 특성 추출
def extract_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # 평균값을 사용하여 차원 축소
    return mfccs

# 데이터 로드 및 MFCC 추출
def load_data(file_paths, labels):
    features = []
    for file_path in file_paths:
        y, sr = load_audio(file_path)
        mfcc = extract_mfcc(y, sr)
        features.append(mfcc)
    return np.array(features), np.array(labels)

# 모델 훈련 및 검증
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return model

# 딥페이크 오디오 탐지
def detect_fake_audio(file_path, model):
    y, sr = load_audio(file_path)
    mfcc = extract_mfcc(y, sr)
    mfcc = mfcc.reshape(1, -1)  # 모델 입력 형태로 변환
    prediction = model.predict(mfcc)
    return prediction[0]

# 예제 데이터 (파일 경로와 레이블)
file_paths = ['real_audio_1.wav', 'fake_audio_1.wav', 'real_audio_2.wav', 'fake_audio_2.wav']
labels = [0, 1, 0, 1]  # 0: 실제 오디오, 1: 딥페이크 오디오

# 데이터 로드
X, y = load_data(file_paths, labels)

# 모델 훈련
model = train_model(X, y)

# 탐지 예제
file_path = 'unknown_audio.wav'
result = detect_fake_audio(file_path, model)
if result == 0:
    print('Real Audio')
else:
    print('Fake Audio')