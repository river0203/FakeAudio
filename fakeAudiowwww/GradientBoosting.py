from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV

def train_and_predict_xgb(X_train, y_train, X_test=None, return_model=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    grid = GridSearchCV(xgb.XGBClassifier(), param_grid, refit=True, verbose=2)
    grid.fit(X_train_scaled, y_train)

    model = grid.best_estimator_

    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(scaler, 'scaler_xgb.pkl')

    if return_model:
        return model, scaler

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return model.predict(X_test_scaled)