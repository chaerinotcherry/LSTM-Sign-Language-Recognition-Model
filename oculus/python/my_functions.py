import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results): #랜드마크를 그려주는 애
    """
    Draw the landmarks on the image.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    # Set the image back to writable mode before drawing
    image.flags.writeable = True

    # Draw landmarks for left hand
    if results.left_hand_landmarks:#왼손 랜드마크를 찾으면 랜드마크를 그린다.
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    # Draw landmarks for right hand
    if results.right_hand_landmarks:#오른손 랜드마크를 찾으면 랜드마크를 그린다.
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    # Draw landmarks for pose (body)
    if results.pose_landmarks:#몸의 랜드마크를 찾으면 랜드마크를 그린다.
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

def image_process(image, model):#랜드마크를 추출하는 함수
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
    """
    # Set the image to read-only mode #읽기 전용으로 바꾸어서 랜드마크를 추출하려고 한다.
    image.flags.writeable = False
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image using the model
    results = model.process(image)
    # Convert the image back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):#랜드마크를 배열에다가 담는 함수.
    """
    Extract the keypoints from the sign landmarks.

    Args:
        results: The processed results containing sign landmarks.

    Returns:
        keypoints (numpy.ndarray): The extracted keypoints.
    """
    # Extract the keypoints for the left hand if present, otherwise set to zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    # Extract the keypoints for the right hand if present, otherwise set to zeros
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    # Extract the keypoints for the pose if present, otherwise set to zeros
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 3)  # Mediapipe Pose model outputs 33 landmarks
    
    # Concatenate the keypoints for both hands and the pose
    keypoints = np.concatenate([lh, rh, pose])
    return keypoints