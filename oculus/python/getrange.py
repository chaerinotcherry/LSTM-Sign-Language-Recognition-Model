import pandas as pd
import glob

# CSV 파일들이 위치한 디렉토리와 파일 패턴
directory = 'C:/Users/VRStudio1/AppData/LocalLow/DefaultCompany/handtracking/'  # 여기에 CSV 파일들이 있는 디렉토리 경로를 입력하세요.
file_pattern = '*.csv'
file_paths = glob.glob(f'{directory}{file_pattern}')

# 결과를 저장할 딕셔너리
results = {}

# 각 파일을 읽어와서 첫 3개의 열의 범위를 계산
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)
        # 첫 3개의 열을 선택
        first_three_cols = df.iloc[2:, :3]
        # 각 열의 범위 계산
        ranges = {}
        for col in first_three_cols.columns:
            col_range = first_three_cols[col].max() - first_three_cols[col].min()
            ranges[col] = col_range
        # 결과를 저장
        results[file_path] = ranges
    except Exception as e:
        print(f"파일 {file_path}를 처리하는 중 오류 발생: {e}")

# 결과를 파일로 저장 (예: 결과를 CSV 파일로 저장)
results_df = pd.DataFrame(results).T
results_df.to_csv('range_results.csv')

print("범위 계산이 완료되었습니다. 결과는 'range_results.csv' 파일에 저장되었습니다.")
