import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_test_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['filename', 'voice_type'], axis=1)
    y = df['voice_type']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.map({'REAL': 0, 'AI': 1})

    return X, y, df['filename']

def predict_model(model, X_test, model_name):
    if model_name == 'CNN':
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions = model.predict(X_test_reshaped)
        return np.argmax(predictions, axis=1)
    elif model_name in ['LSTM', 'RNN']:
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        predictions = model.predict(X_test_reshaped)
        return np.argmax(predictions, axis=1)
    else:
        return model.predict(X_test)

def evaluate_model(model, X_test, y_test, model_name):
    predictions = predict_model(model, X_test, model_name)
    accuracy = (predictions == y_test).mean()
    print(f"{model_name} Model - Test Accuracy: {accuracy:.4f}")
    return predictions

def test_models():
    test_file_path = r"C:\Users\tjdwn\OneDrive\Desktop\AIVoiceFile\testDataCsv\testDataPreprocess.csv"
    X_test, y_test, filenames = load_and_preprocess_test_data(test_file_path)

    models = {
        'CNN': load_model('cnn_model.h5'),
        'LSTM': load_model('lstm_model.h5'),
        'Random Forest': joblib.load('random_forest_model.joblib'),
        'RNN': load_model('rnn_model.h5'),
        'XGBoost': joblib.load('xgboost_model.joblib')
    }

    results = pd.DataFrame({'Filename': filenames})

    for model_name, model in models.items():
        predictions = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = ['AI' if pred == 1 else 'REAL' for pred in predictions]

    # 다수결 투표
    results['Final Result'] = results[list(models.keys())].mode(axis=1)[0]

    # 최종 정확도 계산
    final_accuracy = (results['Final Result'] == y_test.map({0: 'REAL', 1: 'AI'})).mean()
    print(f"Final Majority Voting - Test Accuracy: {final_accuracy:.4f}")

    # 결과를 CSV 파일로 저장
    output_file = 'test_results.csv'
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    test_models()