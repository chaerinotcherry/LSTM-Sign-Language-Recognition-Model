# %%
# Import necessary libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard
import time

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(['water', 'down']) 

# Define the number of sequences and frames to be recorded for each action
sequences = 40
frames = 50

# Set the path where the dataset will be stored
PATH = os.path.join('data')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through each action, sequence, and frame to record data
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        # Read the image from the camera
        _, image = cap.read()

        results = image_process(image, holistic) # 랜드마크를 추출한다.
        image.flags.writeable = True  # Make image writable
        draw_landmarks(image, results) # 랜드마크를 그린다.
        
        # Display text on the image indicating the action and sequence number being recorded
        cv2.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                    (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('Camera', image)
        cv2.waitKey(1)
        
        # Check if the 'q' key was pressed to stop recording
        if keyboard.is_pressed('q'):
            break

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

        # Extract the landmarks from both hands and save them in arrays
        keypoints = keypoint_extraction(results) # 랜드마크들을 저장
        timestamp = time.time()  # 현재 시간을 고유 식별자로 사용
        frame_path = os.path.join(PATH, action, str(sequence), f'{frame}_{timestamp}.npy')
        np.save(frame_path, keypoints)

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
