import os
import pandas as pd
import numpy as np
import librosa
from scipy.fft import fft

# 분석할 폴더 경로
input_folder1 = r'C:\Users\aspp3\OneDrive\바탕 화면\dataSet\real' #real
input_folder2 = r'C:\Users\aspp3\OneDrive\바탕 화면\dataSet\fake' #fake
output_file = r'C:\Users\aspp3\OneDrive\문서\GitHub\FakeAudio'

# 출력 폴더가 없으면 생성
output_folder = os.path.dirname(output_file)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_ifcc(y, sr, n_ifcc=13):
    # FFT 계산
    D = np.abs(librosa.stft(y)) ** 2

    # 멜 필터 뱅크 계산
    n_mels = 40
    mel_basis = librosa.filters.mel(sr=sr, n_fft=D.shape[0] * 2 - 2, n_mels=n_mels)

    # 멜 스펙트럼 계산
    mel_spectrogram = np.dot(mel_basis, D[:mel_basis.shape[1], :])

    # 로그 스펙트럼 계산
    log_spectrogram = librosa.power_to_db(mel_spectrogram)

    # DCT 적용하여 IFCC 계산
    ifcc = librosa.feature.mfcc(S=log_spectrogram, n_mfcc=n_ifcc)
    return ifcc

def extract_features(filename, voice_type):
    # WAV 파일 로드
    y, sr = librosa.load(filename, sr=22050)  # sr=22050을 사용하여 고정된 샘플링 레이트 유지

    # 피치 추출
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    average_pitch = np.mean(pitches)
    pitch_variance = np.var(pitches)

    # 음성 길이 계산
    duration = librosa.get_duration(y=y, sr=sr)

    # FFT 계산
    N = len(y)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * (1 / sr)), N // 2)
    S = 2.0 / N * np.abs(yf[:N // 2])
    noise_level = np.sum(S)

    # MFCCs (Mel-Frequency Cepstral Coefficients) 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)

    # IFCCs (Inverted Frequency Cepstral Coefficients) 추출
    ifccs = extract_ifcc(y, sr, n_ifcc=13)
    ifccs_mean = np.mean(ifccs, axis=1)
    ifccs_var = np.var(ifccs, axis=1)

    features = {
        'filename': os.path.basename(filename),
        'voice_type': voice_type,
        'average_pitch': average_pitch,
        'pitch_variance': pitch_variance,
        'duration': duration,
        'noise_level': noise_level,
    }

    # Add MFCC mean and variance to features
    for i in range(len(mfccs_mean)):
        features[f'mfcc_mean_{i + 1}'] = mfccs_mean[i]
        features[f'mfcc_var_{i + 1}'] = mfccs_var[i]

    # Add IFCC mean and variance to features
    for i in range(len(ifccs_mean)):
        features[f'ifcc_mean_{i + 1}'] = ifccs_mean[i]
        features[f'ifcc_var_{i + 1}'] = ifccs_var[i]

    return features

def analyze_and_save_to_csv(input_folder1, input_folder2, output_file):
    data = []

    files1 = [f for f in os.listdir(input_folder1) if f.endswith('.wav')]
    files2 = [f for f in os.listdir(input_folder2) if f.endswith('.wav')]

    max_len = max(len(files1), len(files2))

    for i in range(max_len):
        if i < len(files1):
            features1 = extract_features(os.path.join(input_folder1, files1[i]), 'AI')
            data.append(features1)
        if i < len(files2):
            features2 = extract_features(os.path.join(input_folder2, files2[i]), 'AI')
            data.append(features2)

    # 데이터프레임 생성 및 CSV 저장
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# 분석 및 CSV 저장 실행
analyze_and_save_to_csv(input_folder1, input_folder2, output_file)
