import pandas as pd


# 경로 설정 전처리 수정
test_csv = r"C:\Users\tjdwn\Downloads\open\test_features.csv"
output_csv = r"C:\Users\tjdwn\Downloads\open\test_features_filtered.csv"

# 필요한 컬럼 이름 (학습 때 사용한 특징들)
needed_columns = [f'mfcc_mean_{i}' for i in range(13)] + ['avg_pitch', 'pitch_var', 'filename']

# CSV 파일 로드
df = pd.read_csv(test_csv)

# 필요한 컬럼만 선택
filtered_df = df[needed_columns]

# CSV 파일로 저장
filtered_df.to_csv(output_csv, index=False)

print(f"Filtered test features saved to {output_csv}")
