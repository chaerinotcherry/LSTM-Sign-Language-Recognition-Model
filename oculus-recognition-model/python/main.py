from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
app = Flask(__name__)

# Load the trained model
model = load_model('best_model.keras')


# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of actions (signs) labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

left_list = []
right_list = []
npy_list = []
predictions = []
last_prediction = ""

# Define the threshold value
threshold = 0.1


def apply_threshold(arr, threshold):
    """Apply threshold to the first three columns."""

    # Compute the range of the first three columns
    ranges = np.ptp(arr[2:, :3], axis=0)  # Peak-to-peak (max - min) along columns

    # Check if all ranges are below the threshold
    if np.all(ranges < threshold):
        arr[:, :] = 0

    return arr


@app.route('/receive_data', methods=['POST'])
def predict():
    global npy_list, predictions, last_prediction, left_list, right_list
    t1 = time.time()
    # Get JSON data
    data = request.get_json()

    # Process the data (assuming 'left' and 'right' keys exist in the JSON)
    left_np = data['leftHand']
    right_np = data['rightHand']

    left_list.append(left_np)
    right_list.append(right_np)

# Check if we have accumulated 50 frames
    if len(left_list) == 50 and len(right_list) == 50:
        t2 = time.time()
        print("50프레임이 쌓일 때까지 걸린 시간은:", t2 - t1)
        # Convert npy_list to a numpy array
        left_keypoints = np.array(left_list)
        right_keypoints = np.array(right_list)

        left_keypoints = apply_threshold(left_keypoints, threshold)
        right_keypoints = apply_threshold(right_keypoints, threshold)

        for idx, (left, right) in enumerate(zip(left_keypoints, right_keypoints), start=1):

            right_np = np.array(right).reshape(-1, 3)
            left_np = np.array(left).reshape(-1, 3)

            # Separate position and rotation parts
            left_np_pos = left_np[0].flatten()
            left_np_rot = left_np[1:].flatten()
            right_np_pos = right_np[0].flatten()
            right_np_rot = right_np[1:].flatten()

            # Min-Max normalization (to avoid ZeroDivisionError)
            def min_max_normalize(arr):
                min_val = np.min(arr)
                max_val = np.max(arr)
                if max_val != min_val:
                    return (arr - min_val) / (max_val - min_val)
                else:
                    return arr - min_val  # Convert all values to 0 if they are the same

            left_np_pos = min_max_normalize(left_np_pos)
            right_np_pos = min_max_normalize(right_np_pos)
            left_np_rot = min_max_normalize(left_np_rot)
            right_np_rot = min_max_normalize(right_np_rot)

            # Reshape the combined array to match the expected input shape
            combined = np.concatenate([left_np_pos, left_np_rot, right_np_pos, right_np_rot])
            npy_list.append(combined)
        npy_array = np.array(npy_list).reshape(1, 50, 114)
        prediction = model.predict(npy_array)
        npy_list.clear()  # Clear list for the next batch of samples

        # Append the prediction to the predictions list
        predictions.append(prediction[0])

        # Clear List for the next set of frames
        left_list.clear()
        right_list.clear()
        npy_list.clear()
        # Calculate the average prediction
        avg_prediction = np.mean(predictions, axis=0)
        print("down:{0} fire:{1} flow:{2} hell:{3} walk:{4} water:{5}" .format(avg_prediction[0],avg_prediction[1],avg_prediction[2],avg_prediction[3],avg_prediction[4],avg_prediction[5]))
        predicted_action_index = np.argmax(avg_prediction)
        predicted_action = actions[predicted_action_index]
        avg_confidence = avg_prediction[predicted_action_index]

        # Clear predictions list
        predictions.clear()

        # Determine the final prediction based on confidence
        if avg_confidence > 0.7:
            last_prediction = predicted_action
        else:
            last_prediction = "Fail"
        t3 = time.time()
        print("데이터를 ai가 판단하는 데 걸린 시간:",t3 - t2)
        # Return the prediction as a JSON response
        return jsonify({'prediction': last_prediction})
    # If less than 50 frames, return a message indicating more data is needed
    return jsonify({})


if __name__ == '__main__':
    app.run(debug=True)
