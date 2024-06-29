import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import GridSearchCV

def train_and_predict_svm(X_train, y_train, X_test=None, return_model=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf']
    }

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train_scaled, y_train)

    model = grid.best_estimator_

    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler_svm.pkl')

    if return_model:
        return model, scaler

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return model.predict(X_test_scaled)
