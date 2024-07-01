import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
from xgboost import XGBClassifier

# 경로 설정 학습 데이터
data_folder = "/Users/iseongjun/Downloads/open/train"
label_csv = r"/Users/iseongjun/Downloads/open/train.csv"

# train.csv 파일 로드 및 라벨링 정보 읽기
label_df = pd.read_csv(label_csv)
file_labels = dict(zip(label_df['id'] + '.ogg', label_df['label']))

# 라벨을 숫자로 변환
label_mapping = {'real': 0, 'fake': 1}
label_df['label'] = label_df['label'].map(label_mapping)


# 데이터 증강 함수
def augment_data(y, sr):
    augmented_data = []
    augmented_data.append(y)
    augmented_data.append(librosa.effects.time_stretch(y, rate=1.2))
    augmented_data.append(librosa.effects.time_stretch(y, rate=0.8))
    augmented_data.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    augmented_data.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    return augmented_data


# 오디오 특징 추출 함수
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    augmented_data = augment_data(y, sr)

    all_features = []
    for y_aug in augmented_data:
        # 특징 추출
        mfccs = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)

        pitches, magnitudes = librosa.core.piptrack(y=y_aug, sr=sr)
        pitches = pitches[pitches > 0]
        if len(pitches) == 0:
            avg_pitch = 0
            pitch_var = 0
        else:
            avg_pitch = np.mean(pitches)
            pitch_var = np.var(pitches)

        # FFT 계산
        N = len(y_aug)
        yf = np.fft.fft(y_aug)
        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / sr)), N // 2)
        S = 2.0 / N * np.abs(yf[:N // 2])
        noise_level = np.sum(S)

        features = np.hstack([mfccs, avg_pitch, pitch_var, noise_level])
        all_features.append(features)

    return np.array(all_features)


# 데이터셋 생성
def create_dataset():
    features = []
    labels = []

    for i, row in tqdm(label_df.iterrows(), total=len(label_df)):
        file_path = os.path.join(data_folder, row['id'] + '.ogg')
        if os.path.exists(file_path):
            feature_vectors = extract_features(file_path)
            for feature_vector in feature_vectors:
                features.append(feature_vector)
                labels.append(row['label'])

    X = np.array(features)
    y = np.array(labels)
    return X, y


# 데이터 로드 및 전처리
X, y = create_dataset()
X = StandardScaler().fit_transform(X)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# XGBoost 모델 클래스
class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# XGBoost 모델 학습
def train_xgboost_model(X_train, y_train):
    model = XGBClassifierWrapper()
    model.fit(X_train, y_train)
    return model


# XGBoost 모델 학습
xgb_model = train_xgboost_model(X_train, y_train)

# 모델 평가
y_pred = xgb_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"XGBoost Model - Test Accuracy: {accuracy:.4f}")

# 모델 저장
xgb_model.model.save_model('xgb_model.json')
print("XGBoost model saved as xgb_model.json")