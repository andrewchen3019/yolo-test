"""
Continuous Flask webcam server with dual YOLO inference + doomscroll detection.
 
Uses two separate model weight files:
  - best.pt  → face detection
  - best1.pt → phone detection
 
If a "phone" bounding box and a "face" bounding box are detected near each
other (both with confidence >= 0.5), a POST is fired to:
  http://localhost:5001/game/doomscrolldetect
 
Usage:
  pip install flask opencv-python ultralytics requests
  python app.py
  Open: http://localhost:5002
"""
 
import os
import time
import threading
 
import cv2
import requests
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
 
# ── macOS camera permission fix ───────────────────────────────────────────────
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"
_warmup = cv2.VideoCapture(0)
_warmup.release()
# ─────────────────────────────────────────────────────────────────────────────
 
# ── Config ────────────────────────────────────────────────────────────────────
FACE_MODEL_PATH  = "best3.pt"    # weights for face detection
PHONE_MODEL_PATH = "best4.pt"   # weights for phone detection
CAMERA_INDEX     = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
 
# Detection
CONFIDENCE       = 0.4          # general inference threshold (all classes)
DOOMSCROLL_CONF  = 0.2         # minimum confidence for phone AND face to trigger
 
# Proximity: centres must be within this fraction of the smaller box's
# diagonal to be considered "near". 0.5 ≈ overlapping / touching.
PROXIMITY_RATIO  = 0.7
 
# Class names — matched against each model's own class list (case-insensitive)
PHONE_CLASSES    = {"phone", "cell phone", "mobile_phone", "smartphone"}
FACE_CLASSES     = {"face", "person", "head"}
 
# Webhook
WEBHOOK_URL      = "http://localhost:5001/game/doomscrolldetect"
COOLDOWN_SEC     = 30
# ─────────────────────────────────────────────────────────────────────────────
 
app        = Flask(__name__)
face_model = YOLO(FACE_MODEL_PATH)
phone_model = YOLO(PHONE_MODEL_PATH)
 
# Shared state
lock          = threading.Lock()
latest_frame  = None
last_trigger  = 0.0
doomscroll_on = False
trigger_start = None
 
# ── Proximity helpers ─────────────────────────────────────────────────────────
 
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)
 
 
def box_diagonal(box):
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
 
 
def boxes_are_near(box_a, box_b, ratio=PROXIMITY_RATIO):
    cx_a, cy_a = box_center(box_a)
    cx_b, cy_b = box_center(box_b)
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    threshold = ratio * min(box_diagonal(box_a), box_diagonal(box_b))
    return dist < threshold
 
 
# ── Helpers to extract qualifying boxes from a result set ─────────────────────
 
def extract_boxes(results, model, target_classes, min_conf):
    """Return xyxy coords for detections matching target_classes above min_conf."""
    boxes = []
    detections = results[0].boxes
    if detections is None:
        return boxes
    for box in detections:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names.get(cls, "").lower()
        if conf >= min_conf and label in target_classes:
            boxes.append(box.xyxy[0].tolist())
    return boxes
 
 
# ── Webhook ───────────────────────────────────────────────────────────────────
 
def fire_webhook():
    try:
        requests.get(WEBHOOK_URL, timeout=2)
        print(f"[doomscroll] fired → {WEBHOOK_URL}")
    except Exception as e:
        print(f"[doomscroll] webhook failed: {e}")
 
 
# ── Main capture + inference loop ─────────────────────────────────────────────
 
def capture_and_infer():
    global latest_frame, last_trigger, doomscroll_on
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
 
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
 
        # ── Run both models independently ─────────────────────────────────────
        face_results  = face_model(frame,  conf=CONFIDENCE, verbose=False)
        phone_results = phone_model(frame, conf=CONFIDENCE, verbose=False)
 
        faces  = extract_boxes(face_results,  face_model,  FACE_CLASSES,  DOOMSCROLL_CONF)
        phones = extract_boxes(phone_results, phone_model, PHONE_CLASSES, DOOMSCROLL_CONF)
 
        # ── Proximity check ───────────────────────────────────────────────────
        triggered = any(
            boxes_are_near(p, f)
            for p in phones
            for f in faces
        )
 
        now = time.time()
        if triggered:
            if trigger_start is None:
                trigger_start = now  # start timing

            # Only fire if it's been continuously true for > 1 second
            if (now - trigger_start) >= 2.0:
                if (now - last_trigger) >= COOLDOWN_SEC:
                    last_trigger = now
                    threading.Thread(target=fire_webhook, daemon=True).start()

            doomscroll_on = True

        else:
            trigger_start = None  # reset timer
            doomscroll_on = False
       
        # ── Draw overlay — merge annotations from both models ─────────────────
        # Start from the face-model annotated frame, then draw phone boxes on top
        annotated = face_results[0].plot()
 
        # Manually draw phone detections from the phone model
        for box in phone_results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = phone_model.names.get(cls, "object")
            color = (0, 165, 255)  # orange for phone boxes
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
 
        bar_color  = (0, 0, 200) if triggered else (0, 150, 0)
        label_text = "DOOMSCROLLING DETECTED!" if doomscroll_on else "No doomscroll"
 
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 38), (0, 0, 0), -1)
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 38), bar_color, 3)
        cv2.putText(annotated, label_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)
 
        _, buffer = cv2.imencode(".jpg", annotated)
        with lock:
            latest_frame = buffer.tobytes()
 
 
# ── MJPEG stream ──────────────────────────────────────────────────────────────
 
def generate_stream():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
 
 
# ── Web UI ────────────────────────────────────────────────────────────────────
 
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Doomscroll Detector</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f0f0f;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: system-ui, sans-serif;
      color: #eee;
      gap: 14px;
    }
    h1  { font-size: 1.4rem; letter-spacing: .05em; }
    img { border: 2px solid #222; border-radius: 8px; max-width: 95vw; }
    .meta { font-size: .78rem; color: #555; }
    .meta span { color: #888; }
  </style>
</head>
<body>
  <h1>📱 Doomscroll Detector</h1>
  <img src="/video_feed" alt="live feed"/>
  <p class="meta">
    Face model: <span>{{ face_model_path }}</span> &nbsp;|&nbsp;
    Phone model: <span>{{ phone_model_path }}</span> &nbsp;|&nbsp;
    Min conf: <span>{{ conf }}</span> &nbsp;|&nbsp;
    Webhook: <span>{{ webhook }}</span> &nbsp;|&nbsp;
    Cooldown: <span>{{ cooldown }}s</span>
  </p>
</body>
</html>
"""
 
 
@app.route("/")
def index():
    return render_template_string(
        PAGE,
        face_model_path=FACE_MODEL_PATH,
        phone_model_path=PHONE_MODEL_PATH,
        conf=DOOMSCROLL_CONF,
        webhook=WEBHOOK_URL,
        cooldown=COOLDOWN_SEC,
    )
 
 
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    t = threading.Thread(target=capture_and_infer, daemon=True)
    t.start()
    print("Server running → http://localhost:5002")
    app.run(host="0.0.0.0", port=5002, debug=False)