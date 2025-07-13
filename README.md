<h1 align="center">🛒 Smart Queue Manager</h1> <p align="center"> Real-time AI-powered retail queue monitoring and alert system using CCTV video analytics. </p> <p align="center"> <a href="#features"><strong>Features</strong></a> · <a href="#tech-stack"><strong>Tech Stack</strong></a> · <a href="#how-it-works"><strong>How It Works</strong></a> · <a href="#getting-started"><strong>Getting Started</strong></a> · <a href="#demo"><strong>Demo</strong></a> </p>

 => Features :

🎥 Upload any CCTV footage or snapshot

🛒 Detect shopping carts using YOLOv8 object detection

📊 Monitor queue length zone-wise in real-time

🚨 Trigger staff alerts when a queue is overcrowded

📁 Support for multiple fake/test scenarios for stress testing

💡 Built-in interactive dashboard with Tailwind CSS

=> Tech Stack : 

-Frontend: HTML, Tailwind CSS, Vanilla JS

-Backend: Python + Flask

-Object Detection: YOLOv8 (Ultralytics)

-Video Processing: OpenCV

-Fake Testing Support: JSON-based fake scenarios

=> How It Works :

1. Upload CCTV Footage
   User uploads a .mp4 video which gets saved and its first frame is extracted.

2. Draw Queue Zones
   Using a separate draw_zones.py, user draws rectangular zones on that frame to define queues.

3. Live Detection Begins
   As the video plays, shopping carts in each zone are detected using YOLOv8 in real-time.

4. Analytics Dashboard
   Shows live cart count zone-wise and highlights detection results visually.

5. Alert System
   If any queue has significantly more carts than others, an automatic alert is triggered.

🔧 Prerequisites :

-Python 3.10+

-Ultralytics YOLOv8

-Flask, OpenCV, Tailwind (CDN)

📦 Installation :

git clone https://github.com/your-username/smart-queue-manager.git
cd smart-queue-manager

pip install -r requirements.txt
python app.py

-Draw zones on the extracted frame using draw_zones.py.

📄 License

This project is for demo and educational purposes.

