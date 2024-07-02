import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# 경로 설정 테스트 데이터
test_data_folder = r"C:\Users\tjdwn\Downloads\open\test"
sample_submission_csv = r"C:\Users\tjdwn\Downloads\open\sample_submission.csv"

# 오디오 특징 추출 함수 (증강 없이)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=32000)
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

# 테스트 데이터 로드 및 특징 추출
def create_test_dataset(test_data_folder):
    features = []
    filenames = []

    for filename in tqdm(os.listdir(test_data_folder), desc="Extracting features"):
        if filename.endswith('.ogg'):
            file_path = os.path.join(test_data_folder, filename)
            feature_vector = extract_features(file_path)
            features.append(feature_vector)
            filenames.append(filename)

    X_test = np.array(features)
    return X_test, filenames

# 테스트 데이터 생성
X_test, filenames = create_test_dataset(test_data_folder)

# 테스트 데이터 저장
test_features_df = pd.DataFrame(X_test, index=filenames)
test_features_df.to_csv('test_features3.csv', index=True)
