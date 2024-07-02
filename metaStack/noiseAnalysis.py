import os
import numpy as np
import pandas as pd
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    n_fft = 512
    hop_length = 256

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Zero-Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Pitch
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    pitches = pitches[pitches > 0]
    if len(pitches) == 0:
        avg_pitch = 0
        pitch_var = 0
    else:
        avg_pitch = np.mean(pitches)
        pitch_var = np.var(pitches)

    features = {
        'mfccs_mean': mfccs_mean,
        'zcr': zcr,
        'spectral_centroid': spectral_centroid,
        'avg_pitch': avg_pitch,
        'pitch_var': pitch_var
    }

    return features

def classify_audio(features):
    # 간단한 기준 설정 (이 기준은 데이터를 바탕으로 조정 가능)
    voice_threshold = 0.02  # 평균 피치의 임계값
    noise_threshold = 0.1   # 스펙트럼 중심의 임계값

    if features['avg_pitch'] > voice_threshold and features['spectral_centroid'] < noise_threshold:
        return "사람 목소리와 소음이 섞여 있음"
    else:
        return "소음만 있음"

def analyze_audio_files(test_data_folder):
    results = []

    for filename in os.listdir(test_data_folder):
        if filename.endswith('.ogg'):
            file_path = os.path.join(test_data_folder, filename)
            features = extract_features(file_path)
            result = classify_audio(features)
            results.append((filename, result))

    return results

# 경로 설정
test_data_folder = r"C:\Users\tjdwn\Downloads\open\test"

# 파일 분석
results = analyze_audio_files(test_data_folder)

# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame(results, columns=['Filename', 'Classification'])

# CSV 파일로 저장
output_csv_path = r"C:\Users\tjdwn\Downloads\open\classification_results.csv"
results_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Results saved to {output_csv_path}")
