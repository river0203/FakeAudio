import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 오디오 데이터 로드
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


# 데이터 증강
def augment_data(y, sr):
    # Noise addition
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise

    # Shifting the sound
    y_roll = np.roll(y, 1600)

    return [y, y_noise, y_roll]


# MFCC 및 추가 특징 추출
def extract_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(contrast.T, axis=0)
    ])

    return features


# 데이터 로드 및 특징 추출
def load_data(folder_path):
    features = []
    labels = []
    for label in ['real', 'fake']:
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for file_name in os.listdir(label_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(label_path, file_name)
                y, sr = load_audio(file_path)
                augmented_data = augment_data(y, sr)
                for y_aug in augmented_data:
                    feature = extract_features(y_aug, sr)
                    features.append(feature)
                    labels.append(0 if label == 'real' else 1)
    return np.array(features), np.array(labels)


# 모델 훈련 및 검증
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Training Accuracy: {accuracy}')
    return model


# 테스트 데이터 로드
def load_test_data(folder_path):
    features = []
    labels = []
    for label in ['real', 'fake']:
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for file_name in os.listdir(label_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(label_path, file_name)
                y, sr = load_audio(file_path)
                feature = extract_features(y, sr)
                features.append(feature)
                labels.append(0 if label == 'real' else 1)
    return np.array(features), np.array(labels)


# 테스트 데이터 정확도 계산
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')
    return accuracy


# 폴더 경로 설정
train_folder_path = 'audio_dataset'  # 'audio_dataset/real' 및 'audio_dataset/fake' 하위 폴더 포함
test_folder_path = 'test_data'  # 'test_data/real' 및 'test_data/fake' 하위 폴더 포함

# 훈련 데이터 로드
X_train, y_train = load_data(train_folder_path)

# 모델 훈련
model = train_model(X_train, y_train)

# 테스트 데이터 로드
X_test, y_test = load_test_data(test_folder_path)

# 모델 평가
evaluate_model(model, X_test, y_test)
