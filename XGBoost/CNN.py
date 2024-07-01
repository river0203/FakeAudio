import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 경로 설정 학습데이터
data_folder = r"C:\Users\tjdwn\Downloads\open\train"
label_csv = r"C:\Users\tjdwn\Downloads\open\train.csv"

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

        features = np.hstack([mfccs, avg_pitch, pitch_var])
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
X = np.expand_dims(X, axis=2)  # CNN 입력을 위해 차원 확장

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 라벨을 one-hot 인코딩
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)


# CNN 모델 생성
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 모델 학습
model = create_cnn_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Model - Test Accuracy: {accuracy:.4f}")

# 모델 저장
model.save('cnn_model.h5')
print("CNN model saved as cnn_model.h5")