import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from sklearn.model_selection import GridSearchCV

def train_and_predict_lgb(X_train, y_train, X_test=None, return_model=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5]
    }

    grid = GridSearchCV(lgb.LGBMClassifier(), param_grid, refit=True, verbose=2, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    model = grid.best_estimator_

    joblib.dump(model, 'lgb_model.pkl')
    joblib.dump(scaler, 'scaler_lgb.pkl')

    if return_model:
        return model, scaler

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return model.predict(X_test_scaled)
