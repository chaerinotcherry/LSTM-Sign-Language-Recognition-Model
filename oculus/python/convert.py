import pandas as pd
import numpy as np
import os

def apply_threshold(df, threshold):
    # Extract the first 3 columns
    df_subset = df.iloc[2:, :3]

    # Compute the range (max - min) for each of the first 3 columns
    ranges = df_subset.max() - df_subset.min()

    # Check if all ranges are less than the threshold
    if np.all(ranges < threshold):
        # Set the values of the first 3 columns to 0
        df.iloc[:, :] = 0

    return df

def process_csv_to_npy(csv_left, csv_right, save_path, sign_language, datetime_str, threshold):
    # Load the CSV files
    df_left = pd.read_csv(csv_left, header=None)
    df_right = pd.read_csv(csv_right, header=None)

    # Apply the threshold
    df_left = apply_threshold(df_left, threshold)
    df_right = apply_threshold(df_right, threshold)

    # Convert DataFrames to lists
    data_left = df_left.values.tolist()
    data_right = df_right.values.tolist()

    # Ensure that both left and right have the same number of rows
    assert len(data_left) == len(data_right), "Left and Right CSV files have different numbers of rows."

    npy_list = []

    # Base directory for saving files
    base_dir = os.path.join(save_path, sign_language + "_" + datetime_str)
    os.makedirs(base_dir, exist_ok=True)

    # Process each list in the large list
    for idx, (left, right) in enumerate(zip(data_left, data_right), start=1):
        # Convert each list to numpy array and reshape
        left_np = np.array(left).reshape(-1, 3)
        right_np = np.array(right).reshape(-1, 3)

        # Separate position and rotation parts
        left_np_pos = left_np[0].flatten()
        left_np_rot = left_np[1:].flatten()
        right_np_pos = right_np[0].flatten()
        right_np_rot = right_np[1:].flatten()

        # Min-Max normalization
        def min_max_normalize(arr):
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val != min_val:
                return (arr - min_val) / (max_val - min_val)
            else:
                return arr - min_val  # Convert all values to 0 if all values are the same

        left_np_pos = min_max_normalize(left_np_pos)
        right_np_pos = min_max_normalize(right_np_pos)
        left_np_rot = min_max_normalize(left_np_rot)
        right_np_rot = min_max_normalize(right_np_rot)

        # Concatenate the left and right arrays
        combined = np.concatenate([left_np_pos, left_np_rot, right_np_pos, right_np_rot])
        npy_list.append(combined)

        # Determine directory for the current file based on idx
        subdir_idx = (idx - 1) // 50
        subdir_path = os.path.join(save_path, f"{subdir_idx}")
        os.makedirs(subdir_path, exist_ok=True)

        npy_file_name = f"{sign_language}_{datetime_str}_{idx}.npy"
        npy_file_path = os.path.join(subdir_path, npy_file_name)

        # Save the combined array as an individual .npy file
        np.save(npy_file_path, combined)

    return np.array(npy_list)

def main():
    # Define the directory containing the CSV files
    csv_directory = 'C:/Users/VRStudio1/AppData/LocalLow/DefaultCompany/handtracking/'

    # Define the threshold value
    threshold = 0.1  # Change this value as needed

    # List all CSV files
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    for csv_file in csv_files:
        # Parse the filename to extract the sign language name, direction, and datetime
        parts = csv_file.split('_')

        sign_language = parts[0]
        direction = parts[1]  # left or right
        datetime_str = parts[2] + '_' + parts[3].split('.')[0]

        # Form paths for left and right CSV files
        if direction == 'left':
            csv_left_path = os.path.join(csv_directory, csv_file)
            csv_right_path = os.path.join(csv_directory, f"{sign_language}_right_{datetime_str}.csv")
            print(csv_left_path)
        else:
            continue

        # Create directory for the sign language if it doesn't exist
        sign_language_dir = os.path.join(csv_directory, sign_language)
        os.makedirs(sign_language_dir, exist_ok=True)

        # Process CSV files and save as .npy
        npy_data = process_csv_to_npy(csv_left_path, csv_right_path, sign_language_dir, sign_language, datetime_str, threshold)

if __name__ == '__main__':
    main()
