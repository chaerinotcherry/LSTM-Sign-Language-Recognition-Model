# %%

# Import necessary libraries
import numpy as np
import os
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import requests
# Flask 서버의 URL 설정
server_url = 'http://3.35.214.173:8501/send-string'

actions = ['water', 'fire','down','flow','hell','walk','fail']

# Load the trained model
model = load_model('my_model.keras')

# Initialize the lists
keypoints = []
predictions = []
last_prediction = ""

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        
        # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
        draw_landmarks(image, results)
        
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints.append(keypoint_extraction(results))
        
        # Check if 30 frames have been accumulated
        if len(keypoints) == 30:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            
            # Append the prediction to the predictions list
            predictions.append(prediction[0])  # Ensure the prediction is added as a 1D array
            
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Calculate the average prediction over the accumulated predictions
            avg_prediction = np.mean(predictions, axis=0)
            predicted_action_index = np.argmax(avg_prediction)
            predicted_action = actions[predicted_action_index]
            avg_confidence = avg_prediction[predicted_action_index]

            # Clear the predictions list for the next set of frames
            predictions = []

            # Check if the average confidence is above the threshold
            if avg_confidence > 0.9:#0.7~0.8 기준으로 하자.
                last_prediction = predicted_action
            else:
                last_prediction = "Fail"
            
            # last_prediction을 Flask 서버로 전송
            try:
                response = requests.post(server_url, json={"prediction": last_prediction})#last_prediction을 request를 보내서 서버에서 보낸 신호를 담기
                if response.status_code == 200:#200은 성공적으로 보냈을때 받는 리턴값
                    print(f'Successfully sent prediction: {last_prediction}')
                else:
                    print(f'Failed to send prediction: {last_prediction}')
            except Exception as e:
                print(f'Error sending prediction: {e}')
        # Show the prediction on the image
        cv2.putText(image, f'Prediction: {last_prediction}', 
                    (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if last_prediction != "Fail" else (0, 0, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)
        
        cv2.waitKey(1)

        # Check if the 'q' key was pressed to stop the loop
        if keyboard.is_pressed(']'):
            break

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break


    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()