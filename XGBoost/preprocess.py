import os
import pandas as pd
import numpy as np
import librosa
from scipy.fft import fft
from tqdm import tqdm

# 분석할 폴더 경로 데이콘 학습데이터
input_folder = r"C:\Users\tjdwn\Downloads\open\train"
output_file = r"C:\Users\tjdwn\Downloads\open\preprocess.csv"
label_csv = r"C:\Users\tjdwn\Downloads\open\train.csv"

# 출력 폴더가 없으면 생성
output_folder = os.path.dirname(output_file)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# train.csv 파일 로드 및 라벨링 정보 읽기
label_df = pd.read_csv(label_csv)
file_labels = dict(zip(label_df['id'] + '.ogg', label_df['label']))


# 변형 데이터 출력
def augment_data(y, sr):
    augmented_data = []
    augmented_data.append(y)
    augmented_data.append(librosa.effects.time_stretch(y, rate=1.2))
    augmented_data.append(librosa.effects.time_stretch(y, rate=0.8))
    augmented_data.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    augmented_data.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    return augmented_data


def extract_features(filename, voice_type):
    y, sr = librosa.load(filename, sr=22050)

    augmented_data = augment_data(y, sr)

    all_features = []
    for y_aug in augmented_data:
        # 피치 추출
        pitches, magnitudes = librosa.core.piptrack(y=y_aug, sr=sr)
        pitches = pitches[pitches > 0]

        if len(pitches) == 0:
            average_pitch = 0
            pitch_variance = 0
        else:
            average_pitch = np.nanmean(pitches)
            pitch_variance = np.nanvar(pitches)

        # FFT 계산
        N = len(y_aug)
        yf = fft(y_aug)
        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / sr)), N // 2)
        S = 2.0 / N * np.abs(yf[:N // 2])

        # MFCCs 추출
        n_fft = min(2048, len(y_aug))
        mfccs = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=13, n_fft=n_fft)
        if np.isnan(mfccs).any():
            mfccs = np.nan_to_num(mfccs)
        mfccs_mean = np.nanmean(mfccs, axis=1)
        mfccs_var = np.nanvar(mfccs, axis=1)

        features = {
            'filename': os.path.basename(filename),
            'voice_type': voice_type,
            'average_pitch': average_pitch,
            'pitch_variance': pitch_variance,
        }

        for i in range(len(mfccs_mean)):
            features[f'mfcc_mean_{i + 1}'] = mfccs_mean[i]
            features[f'mfcc_var_{i + 1}'] = mfccs_var[i]

        all_features.append(features)

    return all_features


def analyze_and_save_to_csv(input_folder, output_file):
    data = []

    files = [f for f in os.listdir(input_folder) if f.endswith('.ogg')]
    for file in tqdm(files, desc=f'Processing {input_folder}'):
        voice_type = file_labels.get(file, 'UNKNOWN')
        features_list = extract_features(os.path.join(input_folder, file), voice_type)
        data.extend(features_list)

    df = pd.DataFrame(data)

    # 데이터셋 무결성 검사: NaN 값이나 무효한 값이 있는지 확인
    if df.isnull().values.any():
        print("Warning: There are NaN values in the dataset.")
        df = df.fillna(0)

    # 데이터프레임 생성 및 CSV 저장
    df.to_csv(output_file, index=False)


# 분석 및 CSV 저장 실행
analyze_and_save_to_csv(input_folder, output_file)
