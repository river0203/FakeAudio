from CNN import train_cnn_model
from LSTM import train_lstm_model
from RandomForest import train_random_forest_model
from RNN import train_rnn_model
from XGBoost import train_xgboost_model


def train_all_models():
    file_path = 'C:/Users/tjdwn/OneDrive/Desktop/AIVoiceFile/preProcess/UpgradePreProcessResult.csv'

    print("Training CNN Model...")
    cnn_model = train_cnn_model(file_path)

    print("\nTraining LSTM Model...")
    lstm_model = train_lstm_model(file_path)

    print("\nTraining Random Forest Model...")
    rf_model = train_random_forest_model(file_path)

    print("\nTraining RNN Model...")
    rnn_model = train_rnn_model(file_path)

    print("\nTraining XGBoost Model...")
    xgb_model = train_xgboost_model(file_path)

    print("\nAll models have been trained successfully.")


if __name__ == "__main__":
    train_all_models()