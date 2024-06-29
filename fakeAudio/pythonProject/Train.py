import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    # Pitch shifting
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

    return [y, y_noise, y_roll, y_pitch]

# MFCC 및 추가 특징 추출
def extract_features(y, sr, n_mfcc=13, fixed_length=50):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < fixed_length:
        pad_width = fixed_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :fixed_length]
    return np.expand_dims(mfccs, axis=-1)

# 데이터 로드 및 특징 추출
def load_data(folder_path, fixed_length=50):
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
                    feature = extract_features(y_aug, sr, fixed_length=fixed_length)
                    features.append(feature)
                    labels.append(0 if label == 'real' else 1)
    return np.array(features), np.array(labels)

# CNN 모델 생성
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 모델 훈련 및 검증
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = create_cnn_model(input_shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Training Accuracy: {accuracy}')

    return model

# 테스트 데이터 로드
def load_test_data(folder_path, fixed_length=50):
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
                feature = extract_features(y, sr, fixed_length=fixed_length)
                features.append(feature)
                labels.append(0 if label == 'real' else 1)
    return np.array(features), np.array(labels)

# 테스트 데이터 정확도 계산
def evaluate_model(model, X_test, y_test):
    y_test = to_categorical(y_test, num_classes=2)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Test Accuracy: {accuracy}')
    return accuracy

# 폴더 경로 설정
train_folder_path = '/content/drive/MyDrive/Data'  # 'audio_dataset/real' 및 'audio_dataset/fake' 하위 폴더 포함
test_folder_path = '/content/drive/MyDrive/testing'  # 'test_data/real' 및 'test_data/fake' 하위 폴더 포함

# 훈련 데이터 로드
X_train, y_train = load_data(train_folder_path)

# 모델 훈련
model = train_model(X_train, y_train)

# 테스트 데이터 로드
X_test, y_test = load_test_data(test_folder_path)

# 모델 평가
evaluate_model(model, X_test, y_test)
