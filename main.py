import cv2
import mediapipe as mp
import subprocess
import numpy as np

# Initialize the mediapipe hand detection module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up the system command to control volume
command = "amixer -D pulse sset Master {}%"

# Initialize the video capture device
cap = cv2.VideoCapture(0)

# Define the volume control parameters
max_volume = 100
min_volume = 0
volume_step = 10

# Initialize the current volume
current_volume = 0

# Set up the hand detection parameters
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2) as hands:

    previous_thumb_to_index_finger_distance = 0

    while cap.isOpened():
        # Read a frame from the video capture device
        ret, frame = cap.read()
        if not ret:
            print("Cannot read a frame from the video stream")
            break

        # Flip the frame horizontally for natural movement
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks in the frame
        results = hands.process(frame_rgb)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the hand landmarks of interest (wrist, thumb, and index finger)
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate the distance between the thumb tip and the index finger tip
                thumb_to_index_finger_distance = ((thumb_landmark.x - index_finger_landmark.x) ** 2 +
                                                  (thumb_landmark.y - index_finger_landmark.y) ** 2) ** 0.5

                # Map the distance to a volume value between 0 and 100
                max_distance = 0.3  # adjust this value to your liking
                min_distance = 0.04
                volume = int(((thumb_to_index_finger_distance - min_distance) / (max_distance - min_distance)) * (max_volume - min_volume) + min_volume)

                # Ensure the volume is within the valid range
                volume = max(min_volume, min(max_volume, volume))

                # Update the current volume
                current_volume = volume

                # Control the system volume using the system command
                subprocess.call(command.format(current_volume), shell=True)

        # Draw the volume bar
        volume_bar_width = int(frame.shape[1] * 0.8)
        volume_bar_height = int(frame.shape[0] * 0.1)
        volume_bar_x = int(frame.shape[1] * 0.1)
        volume_bar_y = int(frame.shape[0] * 0.9)
        cv2.rectangle(frame, (volume_bar_x, volume_bar_y), (volume_bar_x + volume_bar_width, volume_bar_y - volume_bar_height), (255, 255, 255), 2)
        volume_bar_fill_width = int(volume_bar_width * (current_volume / max_volume))
        cv2.rectangle(frame, (volume_bar_x, volume_bar_y), (volume_bar_x + volume_bar_fill_width, volume_bar_y - volume_bar_height), (0, 255, 0), -1)
        cv2.putText(frame, f"Volume: {current_volume}%", (volume_bar_x, volume_bar_y - volume_bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame with hand landmarks and volume bar
        cv2.imshow("Gesture Volume Control", frame)

        # Check for key events
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()