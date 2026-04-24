
import cv2
import os

print(f"OpenCV Version: {cv2.__version__}")

video_path = "/home/kiwi/multi-semantic-searcheng/data/video/videos/video0.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file not found: {video_path}")
    exit(1)

print(f"Testing video: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read a frame. Shape: {frame.shape}")
    else:
        print("Error: Could not read frame.")
    
    cap.release()
