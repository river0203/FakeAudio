import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['filename', 'voice_type'], axis=1)
    y = df['voice_type']

    X = StandardScaler().fit_transform(X)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = to_categorical(y.map({'REAL': 0, 'AI': 1}))

    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape, return_sequences=True),
        SimpleRNN(32),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_rnn_model(file_path):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    model = create_rnn_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"RNN Model - Test Accuracy: {accuracy:.4f}")

    # 모델 저장
    model.save('rnn_model.h5')
    print("RNN model saved as rnn_model.h5")

    return model


if __name__ == "__main__":
    file_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv'
    train_rnn_model(file_path)