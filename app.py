"""
Continuous Flask webcam server with triple YOLO inference + doomscroll detection.
 
Uses three separate model weight files:
  - best.pt  → face detection
  - best1.pt → phone detection
  - best7.pt → hand detection
 
Doomscroll is flagged when a phone, face, AND hand are all detected near each
other (each with confidence >= DOOMSCROLL_CONF) in enough recent frames.
 
Detection robustness improvements:
  - All three models run in PARALLEL threads each cycle (better FPS).
  - Inference only runs every INFER_EVERY_N_FRAMES; cached boxes are drawn
    on intermediate frames so the stream stays smooth.
  - A rolling window vote replaces the brittle "N continuous seconds" timer:
    doomscroll fires when VOTE_THRESHOLD fraction of the last VOTE_WINDOW
    frames were positive. One noisy frame no longer resets the whole counter.
 
Usage:
  pip install flask opencv-python ultralytics requests
  python app.py
  Open: http://localhost:5002
"""
 
import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
 
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
FACE_MODEL_PATH  = "best.pt"    # weights for face detection
PHONE_MODEL_PATH = "best1.pt"   # weights for phone detection
HAND_MODEL_PATH  = "best7.pt"   # weights for hand detection
 
CAMERA_INDEX     = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
 
# Inference is expensive — run it every N frames and reuse cached boxes in
# between. Set to 1 to run every frame (slowest but most responsive).
INFER_EVERY_N_FRAMES = 2
 
# Detection thresholds
CONFIDENCE       = 0.4          # general inference threshold passed to YOLO
DOOMSCROLL_CONF  = 0.2          # minimum conf for a box to count toward doomscroll
 
# Proximity: centres must be within this fraction of the smaller box's
# diagonal to be considered "near". 0.5 ≈ overlapping / touching.
PROXIMITY_RATIO  = 0.7
 
# Class names — matched against each model's own class list (case-insensitive)
PHONE_CLASSES    = {"phone", "cell phone", "mobile_phone", "smartphone"}
FACE_CLASSES     = {"face", "person", "head"}
HAND_CLASSES     = {"hand", "fist", "palm", "open_palm", "closed_fist"}
 
# ── Rolling-window vote ───────────────────────────────────────────────────────
# Keep the last VOTE_WINDOW frame results (True/False).
# Fire when the fraction of True values >= VOTE_THRESHOLD.
# This makes detection robust to a few noisy / missed frames.
VOTE_WINDOW      = 15           # number of recent frames to consider
VOTE_THRESHOLD   = 0.50        # fraction that must be positive to trigger
 
# Webhook
WEBHOOK_URL      = "http://localhost:5001/game/doomscrolldetect"
WEBHOOK_URL_FALSE      = "http://localhost:5001/game/doomscrolldetectfalse"
COOLDOWN_SEC     = 0.5
# ─────────────────────────────────────────────────────────────────────────────
 
app         = Flask(__name__)
face_model  = YOLO(FACE_MODEL_PATH)
phone_model = YOLO(PHONE_MODEL_PATH)
hand_model  = YOLO(HAND_MODEL_PATH)
 
# Shared state (protected by lock)
lock           = threading.Lock()
latest_frame   = None
last_trigger   = 0.0
doomscroll_on  = False
dummy = False
 
# Rolling vote window — deque capped at VOTE_WINDOW length
vote_window    = deque(maxlen=VOTE_WINDOW)
 
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
 
 
# ── Extract qualifying boxes from YOLO results ────────────────────────────────
 
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
 
 
# ── Rolling-window vote logic ─────────────────────────────────────────────────
 
def update_vote(frame_positive: bool) -> bool:
    """
    Append this frame's result to the vote window and return True if the
    current window exceeds the positive-vote threshold.
    """
    vote_window.append(frame_positive)
    if len(vote_window) < VOTE_WINDOW:
        # Not enough history yet — require a full window before firing
        return False
    positive_rate = sum(vote_window) / len(vote_window)
    return positive_rate >= VOTE_THRESHOLD
 
 
# ── Webhook ───────────────────────────────────────────────────────────────────
 
def fire_webhook():
    try:
        requests.get(WEBHOOK_URL, timeout=2)
        print(f"[doomscroll] fired → {WEBHOOK_URL}")
    except Exception as e:
        print(f"[doomscroll] webhook failed: {e}")
 
def fire_webhook_false():
    try:
        requests.get(WEBHOOK_URL_FALSE, timeout=2)
        print(f"[doomscroll] fired → {WEBHOOK_URL_FALSE}")
    except Exception as e:
        print(f"[doomscroll] webhook failed: {e}")
 
# ── Run a single model (used for parallel dispatch) ───────────────────────────
 
def run_model(model, frame, conf):
    return model(frame, conf=conf, verbose=False)
 
 
# ── Main capture + inference loop ─────────────────────────────────────────────
 
def capture_and_infer():
    global latest_frame, last_trigger, doomscroll_on, dummy
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
 
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
 
    # Cached inference outputs — reused on skipped frames
    cached_face_results  = None
    cached_phone_results = None
    cached_hand_results  = None
    frame_counter        = 0
 
    # Thread pool for parallel model inference
    executor = ThreadPoolExecutor(max_workers=3)
 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
 
        frame_counter += 1
        run_inference = (frame_counter % INFER_EVERY_N_FRAMES == 0)
 
        # ── Parallel inference (only on inference frames) ─────────────────────
        if run_inference:
            futures = {
                executor.submit(run_model, face_model,  frame, CONFIDENCE): "face",
                executor.submit(run_model, phone_model, frame, CONFIDENCE): "phone",
                executor.submit(run_model, hand_model,  frame, CONFIDENCE): "hand",
            }
            results_map = {}
            for future in as_completed(futures):
                key = futures[future]
                results_map[key] = future.result()
 
            cached_face_results  = results_map["face"]
            cached_phone_results = results_map["phone"]
            cached_hand_results  = results_map["hand"]
 
        # If we have no results yet (very first frames), skip drawing
        if cached_face_results is None:
            continue
 
        # ── Extract qualifying boxes (always use latest cached results) ───────
        faces  = extract_boxes(cached_face_results,  face_model,  FACE_CLASSES,  DOOMSCROLL_CONF)
        phones = extract_boxes(cached_phone_results, phone_model, PHONE_CLASSES, DOOMSCROLL_CONF)
        hands  = extract_boxes(cached_hand_results,  hand_model,  HAND_CLASSES,  DOOMSCROLL_CONF)
 
        # ── Proximity check: phone near face AND phone near hand ───────────────
        #
        # Logic: we require a phone close to a face (person is looking at it)
        # AND a hand close to that same phone (they are holding/scrolling it).
        # This is a tighter signal than phone+face alone.
        #
        frame_positive = any(
            boxes_are_near(phone, face) or boxes_are_near(phone, hand)
            for phone in phones
            for face  in faces
            for hand  in hands
        )
 
        # ── Rolling-window vote ───────────────────────────────────────────────
        vote_triggered = update_vote(frame_positive)
 
        now = time.time()
        if vote_triggered:
            doomscroll_on = True
            dummy = True
            if (now - last_trigger) >= COOLDOWN_SEC:
                last_trigger = now
                threading.Thread(target=fire_webhook, daemon=True).start()
        elif(dummy):
            doomscroll_on = False
            dummy = False
            threading.Thread(target=fire_webhook_false, daemon=True).start()
        else:
            doomscroll_on = False



 
        # ── Draw overlay ──────────────────────────────────────────────────────
        # Base: face-model annotated frame
        annotated = cached_face_results[0].plot()
 
        # Phone boxes — orange
        for box in cached_phone_results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = phone_model.names.get(cls, "object")
            color = (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
 
        # Hand boxes — cyan
        for box in cached_hand_results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = hand_model.names.get(cls, "hand")
            color = (255, 220, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
 
        # ── Vote meter bar (bottom strip) ─────────────────────────────────────
        vote_ratio   = sum(vote_window) / max(len(vote_window), 1)
        meter_width  = int(FRAME_WIDTH * vote_ratio)
        meter_color  = (0, 200, 0) if vote_ratio < VOTE_THRESHOLD else (0, 0, 220)
        cv2.rectangle(annotated, (0, FRAME_HEIGHT - 8), (meter_width, FRAME_HEIGHT),
                      meter_color, -1)
 
        # ── Status bar (top strip) ────────────────────────────────────────────
        bar_color  = (0, 0, 200) if doomscroll_on else (0, 120, 0)
        label_text = "DOOMSCROLLING DETECTED!" if doomscroll_on else "No doomscroll"
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 38), (0, 0, 0), -1)
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 38), bar_color, 3)
        cv2.putText(annotated, label_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)
 
        # Vote ratio text (top-right corner)
        vote_text = f"vote: {vote_ratio:.0%}"
        cv2.putText(annotated, vote_text, (FRAME_WIDTH - 120, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
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
    .meta { font-size: .78rem; color: #555; text-align: center; line-height: 1.8; }
    .meta span { color: #888; }
  </style>
</head>
<body>
  <h1>📱 Doomscroll Detector</h1>
  <img src="/video_feed" alt="live feed"/>
  <p class="meta">
    Face: <span>{{ face_model_path }}</span> &nbsp;|&nbsp;
    Phone: <span>{{ phone_model_path }}</span> &nbsp;|&nbsp;
    Hand: <span>{{ hand_model_path }}</span><br/>
    Min conf: <span>{{ conf }}</span> &nbsp;|&nbsp;
    Vote window: <span>{{ vote_window }} frames @ {{ vote_threshold }}</span> &nbsp;|&nbsp;
    Infer every: <span>{{ infer_skip }} frames</span> &nbsp;|&nbsp;
    Cooldown: <span>{{ cooldown }}s</span> &nbsp;|&nbsp;
    Webhook: <span>{{ webhook }}</span>
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
        hand_model_path=HAND_MODEL_PATH,
        conf=DOOMSCROLL_CONF,
        vote_window=VOTE_WINDOW,
        vote_threshold=f"{VOTE_THRESHOLD:.0%}",
        infer_skip=INFER_EVERY_N_FRAMES,
        cooldown=COOLDOWN_SEC,
        webhook=WEBHOOK_URL,
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