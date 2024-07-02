import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import joblib

# 데이터 로드
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# GMM 모델 학습
n_components = len(np.unique(y_train))  # 클래스 수를 GMM의 구성 요소 수로 설정
gmm_model = GaussianMixture(n_components=n_components, random_state=42)
gmm_model.fit(X_train)

# GMM 모델의 예측 결과
y_pred_train_gmm = gmm_model.predict(X_train)
y_pred_test_gmm = gmm_model.predict(X_test)

# GMM 모델의 군집 레이블을 실제 레이블과 매핑하여 정확도 계산
def map_cluster_to_label(y_true, y_pred):
    from scipy.stats import mode
    labels = np.zeros_like(y_pred)
    for i in range(n_components):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            continue
        labels[mask] = mode(y_true[mask])[0]
    return labels

# 매핑된 예측 레이블
y_pred_train_mapped = map_cluster_to_label(y_train, y_pred_train_gmm)
y_pred_test_mapped = map_cluster_to_label(y_test, y_pred_test_gmm)

# 정확도 계산
train_accuracy = accuracy_score(y_train, y_pred_train_mapped)
test_accuracy = accuracy_score(y_test, y_pred_test_mapped)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# 학습된 모델 저장
joblib.dump(gmm_model, 'gmm_model.pkl')

# 예측 결과 저장
pd.DataFrame(y_pred_train_mapped, columns=['GMM']).to_csv('gmm_predictions_train.csv', index=False)
pd.DataFrame(y_pred_test_mapped, columns=['GMM']).to_csv('gmm_predictions_test.csv', index=False)
