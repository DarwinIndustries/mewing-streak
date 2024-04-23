import cv2
import dlib
import numpy as np
import subprocess
from imutils import face_utils

# Define constants
MOUTH_AR_THRESH = 0.79

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the two sets of vertical mouth landmarks
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])   # 53, 57

    # Compute the euclidean distance between the horizontal mouth landmarks
    C = np.linalg.norm(mouth[0] - mouth[6])   # 49, 55

    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # Return the mouth aspect ratio
    return mar

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the mouth coordinates
        mouth = shape[49:68]

        # Compute the mouth aspect ratio
        mar = mouth_aspect_ratio(mouth)

        # Compute the convex hull for the mouth and visualize it
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Display the MAR value in the top left corner
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display "MOUTH OPEN" if the mouth is open (MAR exceeds the threshold)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "MOUTH OPEN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Shut down the Macbook if the mouth is detected as open
            subprocess.run(['sudo', 'shutdown', '-h', 'now'])

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
