import os
import glob
import datetime

# 절대 경로 설정
directory_path = r'C:\Users\silve\OneDrive\바탕 화면\skku\Sign-Language-Translator\data'  # 실제 경로로 수정

# 디렉터리 경로 확인
print(f"Directory path: {os.path.abspath(directory_path)}")
print(f"Contents of directory: {os.listdir(directory_path)}")

# 삭제할 날짜 설정 (7월 22일)
target_date = '2024-07-22'

# directory_path 안의 모든 .npy 파일을 검색합니다.
file_paths = glob.glob(os.path.join(directory_path, '**', '*.npy'), recursive=True)

# 디버깅: 파일 경로 출력
print(f"Found {len(file_paths)} .npy files.")

# 타겟 날짜의 파일을 삭제합니다.
for file_path in file_paths:
    modified_time = os.path.getmtime(file_path)
    file_modified_date = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d')
    print(f"File: {file_path}, Modified Date: {file_modified_date}")  # 디버깅: 파일과 수정 날짜 출력
    if file_modified_date == target_date:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {file_path}, {e}")

print("Deletion complete.")