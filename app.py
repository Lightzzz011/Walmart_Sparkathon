from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2, json, os
import random
from threading import Lock

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("zones", exist_ok=True)

# ✅ Shared state for real-time counts
latest_counts = {}
count_lock = Lock()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    original_filename = file.filename
    video_path = os.path.join("uploads", original_filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Failed to extract frame"}), 500

    frame_name = f"frame_{os.path.splitext(original_filename)[0]}.jpg"
    frame_path = os.path.join("static", frame_name)
    cv2.imwrite(frame_path, frame)

    return jsonify({
        "message": f"{original_filename} uploaded. Now draw zones.",
        "frame_path": f"/static/{frame_name}",
        "video_name": original_filename
    })

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    file_path = "temp.jpg"
    file.save(file_path)

    results = model(file_path)
    results[0].save(filename="static/output.jpg")

    count = sum(1 for box in results[0].boxes
                if model.names[int(box.cls[0])] == "shopping-trolley")

    return jsonify({
        "cart_count": count,
        "image_path": "static/output.jpg"
    })

@app.route('/save-zones', methods=['POST'])
def save_zones():
    data = request.get_json()
    video_name = data.get('video_name', 'default')
    zones = data.get('zones', [])
    zone_file = f'zones/zones_{os.path.splitext(video_name)[0]}.json'

    with open(zone_file, 'w') as f:
        json.dump(zones, f, indent=2)

    return jsonify({"message": f"Zones saved to {zone_file}"})


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/video-stream/<video_name>")
def video_stream(video_name):
    video_path = f"uploads/{video_name}"
    zone_path = f"zones/zones_{os.path.splitext(video_name)[0]}.json"

    if not os.path.exists(video_path) or not os.path.exists(zone_path):
        return "Video or zone file not found", 404

    def gen_frames():
        cap = cv2.VideoCapture(video_path)
        with open(zone_path, "r") as f:
            zones = json.load(f)

        def compute_iou(boxA, boxB):
            xA = max(boxA["x1"], boxB["x1"])
            yA = max(boxA["y1"], boxB["y1"])
            xB = min(boxA["x2"], boxB["x2"])
            yB = min(boxA["y2"], boxB["y2"])
            inter = max(0, xB - xA) * max(0, yB - yA)
            areaA = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
            areaB = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
            return inter / (areaA + areaB - inter + 1e-6)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 384))
            result = model.predict(frame, conf=0.25)[0]
            cart_counts = [0] * len(zones)

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if "trolley" not in class_name.lower() and "cart" not in class_name.lower():
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cart_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

                for i, zone in enumerate(zones):
                    if compute_iou(cart_box, zone) > 0.01:
                        cart_counts[i] += 1
                        break

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            for i, zone in enumerate(zones):
                cv2.rectangle(frame, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), (0, 255, 0), 2)
                cv2.putText(frame, f"Zone {i+1}: {cart_counts[i]}", (zone["x1"], zone["y1"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ✅ Update shared latest cart counts
            with count_lock:
                latest_counts[video_name] = {f"Zone {i+1}": cart_counts[i] for i in range(len(zones))}

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/realtime-analytics")
def realtime_analytics():
    video = request.args.get("video")
    with count_lock:
        data = latest_counts.get(video)
    if not data:
        return jsonify({"error": "No live data available"}), 404
    return jsonify(data)


@app.route('/alerts')
def alerts():
    try:
        video_name = request.args.get("video")  # e.g., FakeVid2.mp4
        if not video_name:
            return jsonify({"alert": ["❗ No video selected."]})

        # 🔥 TEMP: Fake cart counts for testing scenarios
        fake_scenarios = {
            "FakeVid1.mp4": {"Zone 1": 2, "Zone 2": 2, "Zone 3": 7},
            "FakeVid2.mp4": {"Zone 1": 5, "Zone 2": 2, "Zone 3": 2},
            "FakeVid3.mp4": {"Zone 1": 7, "Zone 2": 12, "Zone 3": 6},
        }

        cart_data = fake_scenarios.get(video_name)
        if not cart_data:
            return jsonify({"alert": ["⚠️ No fake scenario found for this video."]})

        avg = sum(cart_data.values()) / len(cart_data)
        alerts = []

        for zone, count in cart_data.items():
            if count > avg + 2:
                alerts.append(f"🚨 {zone} is overcrowded. Please redirect staff.")

        if not alerts:
            alerts.append("✅ All queues are balanced.")

        return jsonify({"alert": alerts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True)
