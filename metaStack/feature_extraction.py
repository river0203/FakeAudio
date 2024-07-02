import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import random

# 경로 설정
train_data_folder = r"C:\Users\tjdwn\Downloads\open\train"
train_features_csv = r"C:\Users\tjdwn\Downloads\open\train_features_augmented.csv"


# 오디오 특징 추출 함수
def extract_features(y, sr):
    if np.mean(np.abs(y)) < 1e-5:  # 소리가 없는 경우
        return np.zeros(28)  # 빈 특징 벡터 반환

    # MFCC 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # LFCC 추출
    lfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    lfccs_mean = np.mean(lfccs.T, axis=0)

    # Pitch 계산
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) == 0:
        avg_pitch = 0
        pitch_var = 0
    else:
        avg_pitch = np.mean(pitches)
        pitch_var = np.var(pitches)

    features = np.hstack([mfccs_mean, lfccs_mean, avg_pitch, pitch_var])
    return features


# 데이터 증강 함수
def augment_audio(y, sr):
    # 시간 스트레칭
    y_stretch = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))

    # 피치 시프트
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))

    return y_stretch, y_pitch


# 학습 데이터 로드, 특징 추출 및 증강
def create_train_dataset(train_data_folder):
    features = []
    filenames = []
    labels = []

    for filename in tqdm(os.listdir(train_data_folder), desc="Processing files"):
        if filename.endswith('.ogg'):
            file_path = os.path.join(train_data_folder, filename)
            y, sr = librosa.load(file_path, sr=32000)

            # 원본 데이터 특징 추출
            feature_vector = extract_features(y, sr)
            features.append(feature_vector)
            filenames.append(filename)
            labels.append('real' if 'real' in filename else 'fake')

            # 데이터 증강
            y_stretch, y_pitch = augment_audio(y, sr)

            # 증강된 데이터 특징 추출
            features.append(extract_features(y_stretch, sr))
            filenames.append(f"aug1_{filename}")
            labels.append('real' if 'real' in filename else 'fake')

            features.append(extract_features(y_pitch, sr))
            filenames.append(f"aug2_{filename}")
            labels.append('real' if 'real' in filename else 'fake')

    X_train = np.array(features)
    return X_train, filenames, labels


# 학습 데이터 생성
X_train, filenames, labels = create_train_dataset(train_data_folder)

# 학습 데이터 저장
train_features_df = pd.DataFrame(X_train, index=filenames)
train_features_df['label'] = labels
train_features_df.to_csv(train_features_csv, index=True)

print(f"Train features saved to {train_features_csv}")
print(f"Total samples: {len(filenames)}")
print(f"Original samples: {len(os.listdir(train_data_folder))}")
print(f"Augmented samples: {len(filenames) - len(os.listdir(train_data_folder))}")