import cv2
import os

# === CONFIG ===
video_name = "QueueVid1.mp4"  # ⬅️ Change this for each video
video_path = os.path.join("uploads", video_name)

# === Save Frame ===
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()

if success:
    frame_filename = f"static/frame_{os.path.splitext(video_name)[0]}.jpg"
    cv2.imwrite(frame_filename, frame)
    print(f"✅ First frame saved as {frame_filename}")
else:
    print("❌ Failed to read video.")

cap.release()
