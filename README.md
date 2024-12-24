## Prerequisites
Ensure you have Python installed and the following libraries:

OpenCV
MediaPipe
Install Dependencies


## Run:
``pip install opencv-python mediapipe``

## How to Use
1. For Real-Time Detection
Connect a webcam to your computer.
Update the cv2.VideoCapture line in the script to:
``cap = cv2.VideoCapture(0)``

Run the script:
``python main.py``

3. For Pre-Recorded Videos
Place your video file (e.g., exercise_video.mp4) in the project directory.
Update the video_path variable in the script:
``video_path = "exercise_video.mp4"``

Run the script:
``python main.py``
