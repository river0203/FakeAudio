import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['filename', 'voice_type'], axis=1)
    y = df['voice_type']

    X = StandardScaler().fit_transform(X)
    y = y.map({'REAL': 0, 'AI': 1})

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_xgboost_model(file_path):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model - Test Accuracy: {accuracy:.4f}")

    # 모델 저장
    joblib.dump(model, 'xgboost_model.joblib')
    print("XGBoost model saved as xgboost_model.joblib")

    return model


if __name__ == "__main__":
    file_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv'
    train_xgboost_model(file_path)