import cv2
import json, os
from ultralytics import YOLO

# === Load YOLOv8 model ===
model = YOLO("best.pt")

# === Class List for Debugging ===
print("\n🔎 Model class names:")
print(model.names)

# === Input Video ===
video_name = input("🎥 Enter video filename (inside uploads/): ").strip()
video_path = os.path.join("uploads", video_name)

# === Load Matching Zone File ===
zone_file = f"zones/zones_{os.path.splitext(video_name)[0]}.json"
if not os.path.exists(zone_file):
    print(f"❌ Zone file not found: {zone_file}")
    exit()

with open(zone_file, "r") as f:
    zones = json.load(f)
print(f"✅ Loaded zones from {zone_file}")

# === IoU Function ===
def compute_iou(boxA, boxB):
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    boxB_area = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])

    return inter_area / (boxA_area + boxB_area - inter_area + 1e-6)

# === Start Video Processing ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (640, 384))  # Match zones and speed
    result = model.predict(frame, conf=0.25)[0]

    cart_counts = [0] * len(zones)

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Filter only trolley/cart classes
        if "trolley" not in class_name.lower() and "cart" not in class_name.lower():
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cart_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        matched = False
        for i, zone in enumerate(zones):
            iou = compute_iou(cart_box, zone)
            if iou > 0.01:
                cart_counts[i] += 1
                matched = True
                break

        # Draw cart box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        symbol = "✅" if matched else "❌"
        cv2.putText(frame, symbol, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if matched else (0, 0, 255), 2)

    # Draw zones
    for i, zone in enumerate(zones):
        cv2.rectangle(frame, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), (0, 255, 0), 2)
        cv2.putText(frame, f"Zone {i+1}: {cart_counts[i]}", (zone["x1"], zone["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Smart Queue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
