import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# 경로 설정 테스트 데이터
test_folder = r"C:\Users\tjdwn\Downloads\open\test"
output_csv = r"C:\Users\tjdwn\Downloads\open\test_features.csv"


# 오디오 특징 추출 함수
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # 특징 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) == 0:
        avg_pitch = 0
        pitch_var = 0
    else:
        avg_pitch = np.mean(pitches)
        pitch_var = np.var(pitches)

    # FFT 계산
    N = len(y)
    yf = np.fft.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * (1 / sr)), N // 2)
    S = 2.0 / N * np.abs(yf[:N // 2])
    noise_level = np.sum(S)

    features = np.hstack([mfccs_mean, avg_pitch, pitch_var, noise_level])
    return features


# 테스트 데이터셋 생성
def create_test_dataset():
    features = []
    filenames = []

    for file_name in tqdm(os.listdir(test_folder)):
        if file_name.endswith('.ogg'):
            file_path = os.path.join(test_folder, file_name)
            if os.path.exists(file_path):
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                filenames.append(file_name)

    X = np.array(features)
    return X, filenames


# 데이터 로드 및 전처리
X_test, test_filenames = create_test_dataset()

# 특징을 DataFrame으로 변환
feature_names = [f'mfcc_mean_{i}' for i in range(13)] + ['avg_pitch', 'pitch_var', 'noise_level']
df = pd.DataFrame(X_test, columns=feature_names)
df['filename'] = test_filenames

# CSV 파일로 저장
df.to_csv(output_csv, index=False)

print(f"Test features saved to {output_csv}")
