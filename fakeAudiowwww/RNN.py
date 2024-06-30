import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['filename', 'voice_type'], axis=1)
    y = df['voice_type']

    X = StandardScaler().fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = to_categorical(y.map({'REAL': 0, 'AI': 1}))

    return train_test_split(X, y, test_size=0.2, random_state=42)


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


def train_cnn_model(file_path):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    model = create_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Model - Test Accuracy: {accuracy:.4f}")

    return model


if __name__ == "__main__":
    file_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv'
    train_cnn_model(file_path)