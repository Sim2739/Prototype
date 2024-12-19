import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize counters and state
squat_count = 0
squat_state = "UP"

# Load video file
video_path = "squat.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Scaling factors for resizing
scale_width = 0.5  # Adjust width
scale_height = 0.4  # Adjust height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Process pose
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the video frame
    frame_height, frame_width = frame.shape[:2]
    new_width = int(frame_width * scale_width)
    new_height = int(frame_height * scale_height)
    resized_frame = cv2.resize(image, (new_width, new_height))

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(resized_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for hips, knees, and ankles
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angles
        knee_angle = calculate_angle(hip, knee, ankle)

        # Detect squats based on knee angle
        if knee_angle < 70:  # Squat position (angle less than 70 degrees)
            if squat_state == "UP":
                squat_state = "DOWN"  # Transition to squat
        elif knee_angle > 160:  # Standing position (angle greater than 160 degrees)
            if squat_state == "DOWN":
                squat_state = "UP"  # Transition to standing
                squat_count += 1  # Increment squat count

        # Display knee angle, squat count, and state
        cv2.putText(resized_frame, f"Knee Angle: {int(knee_angle)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resized_frame, f"Squats: {squat_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(resized_frame, f"State: {squat_state}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resized frame
    cv2.imshow('Squat Detection', resized_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
