import cv2
import json
import os

# === Setup output folder first ===
os.makedirs("zones", exist_ok=True)

# === Ask for video ===
video_name = input("🎥 Enter video filename (inside uploads/): ").strip()
video_path = os.path.join("uploads", video_name)

if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    exit()

frame_path = f"static/frame_{os.path.splitext(video_name)[0]}.jpg"
zones_output = f"zones/zones_{os.path.splitext(video_name)[0]}.json"

# === Extract or load frame ===
if not os.path.exists(frame_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        frame = cv2.resize(frame, (640, 384))
        cv2.imwrite(frame_path, frame)
        print(f"✅ Frame saved to {frame_path}")
    else:
        print("❌ Couldn't extract frame")
        exit()
else:
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, (640, 384))
    print(f"✅ Loaded frame from {frame_path}")

frame_copy = frame.copy()

# === Drawing logic ===
zones = []
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, zones, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zones.append({"x1": ix, "y1": iy, "x2": x, "y2": y})
        cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        print(f"🟩 Zone added: ({ix}, {iy}) to ({x}, {y})")

# === Open window ===
cv2.namedWindow("Draw Zones")
cv2.setMouseCallback("Draw Zones", draw_rectangle)

print("✏️ Draw zones with mouse. Press 'q' to finish and save.")

while True:
    cv2.imshow("Draw Zones", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# === Save only if zones exist ===
if zones:
    with open(zones_output, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"✅ Zones saved to {zones_output}")
else:
    print("⚠️ No zones drawn. Nothing saved.")
