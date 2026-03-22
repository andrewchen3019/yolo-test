"""
Continuous Flask webcam server with triple YOLO inference + doomscroll detection.
Hailo edition — all three models run on the Hailo-8L via .hef files.

Prerequisites (on the Pi):
    sudo apt install hailo-all
    pip install flask opencv-python numpy requests

HEF files expected in the same directory as this script:
    best_face.hef   — custom face model
    yolo12n.hef     — YOLO12n compiled for phone detection (cell phone class = 67)
    best_hand.hef   — custom hand model

Open: http://<pi-ip>:5002
"""

import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template_string

from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"

# ── Config ────────────────────────────────────────────────────────────────────
FACE_HEF_PATH  = "best_face.hef"
PHONE_HEF_PATH = "yolo12n.hef"
HAND_HEF_PATH  = "best_hand.hef"

# COCO class index 67 = "cell phone" — only used for the phone model
PHONE_CLASS_ID   = 67
PHONE_CLASS_NAME = "cell phone"

CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

INFER_EVERY_N_FRAMES = 6

CONFIDENCE      = 0.4   # minimum score for drawing boxes
DOOMSCROLL_CONF = 0.2   # minimum score to count toward doomscroll logic

PROXIMITY_RATIO = 0.7

# Class name sets — matched against names dicts defined per model below
FACE_CLASSES  = {"face", "person", "head"}
HAND_CLASSES  = {"hand", "fist", "palm", "open_palm", "closed_fist"}

VOTE_WINDOW    = 15
VOTE_THRESHOLD = 0.50

WEBHOOK_URL       = "http://localhost:5001/game/doomscrolldetect"
WEBHOOK_URL_FALSE = "http://localhost:5001/game/doomscrolldetectfalse"
COOLDOWN_SEC      = 0.5
# ─────────────────────────────────────────────────────────────────────────────


# ── Hailo model wrapper ───────────────────────────────────────────────────────

class HailoModel:
    """
    Wraps a single .hef model for synchronous inference.

    The VDevice and network group are initialised once at startup.
    InferVStreams is opened fresh each call — this is safe and avoids
    the complexity of sharing a pipeline across threads.

    names: dict mapping int class index → str class name.
           For COCO models supply the full 80-class dict or a subset.
           For custom models supply whatever your training used.
    """

    def __init__(self, hef_path: str, names: dict):
        self.hef_path = hef_path
        self.names    = names

        self.hef    = HEF(hef_path)
        self.target = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group        = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        input_info        = self.hef.get_input_vstream_infos()[0]
        self.input_name   = input_info.name
        self.input_h, self.input_w, _ = input_info.shape

        # UINT8 input (0-255 BGR→RGB) — avoids float normalisation on the Pi CPU
        self.input_params  = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        self.output_infos = self.hef.get_output_vstream_infos()

        print(f"[Hailo] Loaded {os.path.basename(hef_path)} "
              f"  input={self.input_name} {self.input_h}×{self.input_w}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize → RGB → uint8, add batch dim."""
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    def __call__(self, frame: np.ndarray, conf: float = CONFIDENCE,
                 classes: list = None):
        """
        Run inference on one BGR frame.
        Returns a list of dicts:
            [{"box": [x1,y1,x2,y2], "conf": float, "cls": int, "label": str}, ...]
        Coordinates are absolute pixels in the ORIGINAL frame's space.
        """
        orig_h, orig_w = frame.shape[:2]
        preprocessed   = self.preprocess(frame)
        input_data      = {self.input_name: np.expand_dims(preprocessed, 0)}

        with InferVStreams(self.network_group,
                          self.input_params,
                          self.output_params) as pipeline:
            with self.network_group.activate(self.network_group_params):
                raw = pipeline.infer(input_data)

        return self._postprocess(raw, orig_w, orig_h, conf, classes)

    def _postprocess(self, raw: dict, orig_w: int, orig_h: int,
                     min_conf: float, filter_classes: list):
        """
        Decode Hailo YOLO NMS output.

        Hailo Model Zoo compiles YOLO with NMS baked in.  The merged output
        tensor has shape (1, N, 6) where each row is:
            [y_min, x_min, y_max, x_max, score, class_id]  — normalised 0-1.

        If your compilation produced un-merged per-class tensors (older DFC
        versions), set HAILO_MERGED_NMS = False below and adjust accordingly.
        """
        HAILO_MERGED_NMS = True   # ← set False for multi-tensor NMS output

        detections = []

        if HAILO_MERGED_NMS:
            # Collect the single merged output tensor
            tensor = None
            for name, arr in raw.items():
                arr = np.array(arr)
                if arr.ndim == 3 and arr.shape[-1] == 6:
                    tensor = arr[0]   # shape (N, 6)
                    break
            if tensor is None:
                # Fallback: flatten whatever came back
                for arr in raw.values():
                    arr = np.array(arr).reshape(-1, 6)
                    tensor = arr
                    break

            for row in tensor:
                y1, x1, y2, x2, score, cls_id = row
                cls_id = int(cls_id)
                if score < min_conf:
                    continue
                if filter_classes is not None and cls_id not in filter_classes:
                    continue
                label = self.names.get(cls_id, str(cls_id))
                # De-normalise to original frame pixels
                detections.append({
                    "box":   [x1 * orig_w, y1 * orig_h,
                              x2 * orig_w, y2 * orig_h],
                    "conf":  float(score),
                    "cls":   cls_id,
                    "label": label,
                })
        else:
            # Per-class tensor layout: each output is (1, num_boxes_for_class, 5)
            # where 5 = [y1, x1, y2, x2, score], class given by tensor name.
            for out_info in self.output_infos:
                arr     = np.array(raw[out_info.name])[0]   # (M, 5)
                cls_id  = int(out_info.name.split("_")[-1]) # depends on naming
                label   = self.names.get(cls_id, str(cls_id))
                if filter_classes is not None and cls_id not in filter_classes:
                    continue
                for row in arr:
                    y1, x1, y2, x2, score = row
                    if score < min_conf:
                        continue
                    detections.append({
                        "box":   [x1 * orig_w, y1 * orig_h,
                                  x2 * orig_w, y2 * orig_h],
                        "conf":  float(score),
                        "cls":   cls_id,
                        "label": label,
                    })

        return detections


# ── COCO names (subset — only what we use) ────────────────────────────────────
# Full 80-class list would go here; we only need index 67 for inference but
# keeping a reasonable subset avoids key errors if other classes slip through.
COCO_NAMES = {
    0: "person",    24: "backpack", 26: "handbag",
    41: "cup",      42: "fork",     43: "knife",
    44: "spoon",    45: "bowl",     46: "banana",
    63: "laptop",   64: "mouse",    65: "remote",
    66: "keyboard", 67: "cell phone",
    73: "book",     74: "clock",    76: "scissors",
}

# Custom model class names — update these to match your training labels exactly.
# Indices must match the class indices used during training.
FACE_NAMES = {0: "face"}          # adjust if your model has more classes
HAND_NAMES = {0: "hand"}          # adjust if your model has more classes


# ── Load models ───────────────────────────────────────────────────────────────
print("[startup] Loading Hailo models …")
face_model  = HailoModel(FACE_HEF_PATH,  names=FACE_NAMES)
phone_model = HailoModel(PHONE_HEF_PATH, names=COCO_NAMES)
hand_model  = HailoModel(HAND_HEF_PATH,  names=HAND_NAMES)
print("[startup] All models ready.")


# ── Shared state ──────────────────────────────────────────────────────────────
app           = Flask(__name__)
lock          = threading.Lock()
latest_frame  = None
last_trigger  = 0.0
doomscroll_on = False
dummy         = False
vote_window   = deque(maxlen=VOTE_WINDOW)


# ── Geometry helpers ──────────────────────────────────────────────────────────

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


# ── Rolling-window vote ───────────────────────────────────────────────────────

def update_vote(frame_positive: bool) -> bool:
    vote_window.append(frame_positive)
    if len(vote_window) < VOTE_WINDOW:
        return False
    return (sum(vote_window) / len(vote_window)) >= VOTE_THRESHOLD


# ── Webhooks ──────────────────────────────────────────────────────────────────

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


# ── Draw helpers ──────────────────────────────────────────────────────────────

def draw_detections(frame, detections, color):
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        label = f"{d['label']} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ── Main capture + inference loop ─────────────────────────────────────────────

def run_model_thread(model, frame, conf, classes=None):
    return model(frame, conf=conf, classes=classes)


def capture_and_infer():
    global latest_frame, last_trigger, doomscroll_on, dummy

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    cached_faces  = []
    cached_phones = []
    cached_hands  = []
    frame_counter = 0

    executor = ThreadPoolExecutor(max_workers=3)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counter += 1
        run_inference = (frame_counter % INFER_EVERY_N_FRAMES == 0)

        if run_inference:
            futures = {
                executor.submit(run_model_thread, face_model,  frame, DOOMSCROLL_CONF):                        "face",
                executor.submit(run_model_thread, phone_model, frame, DOOMSCROLL_CONF, [PHONE_CLASS_ID]):       "phone",
                executor.submit(run_model_thread, hand_model,  frame, DOOMSCROLL_CONF):                        "hand",
            }
            results_map = {}
            for future in as_completed(futures):
                results_map[futures[future]] = future.result()

            # Filter to only classes we care about
            cached_faces  = [d for d in results_map["face"]
                             if d["label"].lower() in FACE_CLASSES]
            cached_phones = results_map["phone"]   # already filtered to class 67
            cached_hands  = [d for d in results_map["hand"]
                             if d["label"].lower() in HAND_CLASSES]

        # ── Proximity / doomscroll ────────────────────────────────────────────
        face_boxes  = [d["box"] for d in cached_faces]
        phone_boxes = [d["box"] for d in cached_phones]
        hand_boxes  = [d["box"] for d in cached_hands]

        frame_positive = any(
            boxes_are_near(phone, face) or boxes_are_near(phone, hand)
            for phone in phone_boxes
            for face  in face_boxes
            for hand  in hand_boxes
        )

        vote_triggered = update_vote(frame_positive)
        now = time.time()

        if vote_triggered:
            doomscroll_on = True
            dummy = True
            if (now - last_trigger) >= COOLDOWN_SEC:
                last_trigger = now
                threading.Thread(target=fire_webhook, daemon=True).start()
        elif dummy:
            doomscroll_on = False
            dummy = False
            threading.Thread(target=fire_webhook_false, daemon=True).start()
        else:
            doomscroll_on = False

        # ── Draw ──────────────────────────────────────────────────────────────
        annotated = frame.copy()
        draw_detections(annotated, cached_faces,  (0, 255, 0))      # green  — face
        draw_detections(annotated, cached_phones, (0, 165, 255))    # orange — phone
        draw_detections(annotated, cached_hands,  (255, 220, 0))    # yellow — hand

        # Vote meter (bottom strip)
        vote_ratio  = sum(vote_window) / max(len(vote_window), 1)
        meter_width = int(FRAME_WIDTH * vote_ratio)
        meter_color = (0, 200, 0) if vote_ratio < VOTE_THRESHOLD else (0, 0, 220)
        cv2.rectangle(annotated, (0, FRAME_HEIGHT - 8), (meter_width, FRAME_HEIGHT),
                      meter_color, -1)

        # Status bar (top strip)
        bar_color  = (0, 0, 200) if doomscroll_on else (0, 120, 0)
        label_text = "DOOMSCROLLING DETECTED!" if doomscroll_on else "No doomscroll"
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 38), (0, 0, 0), -1)
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 38), bar_color, 3)
        cv2.putText(annotated, label_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        vote_text = f"vote: {vote_ratio:.0%}"
        cv2.putText(annotated, vote_text, (FRAME_WIDTH - 120, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

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
    Face: <span>{{ face_hef }}</span> &nbsp;|&nbsp;
    Phone: <span>{{ phone_hef }} (cell phone only)</span> &nbsp;|&nbsp;
    Hand: <span>{{ hand_hef }}</span><br/>
    Min conf: <span>{{ conf }}</span> &nbsp;|&nbsp;
    Vote window: <span>{{ vote_window }} frames @ {{ vote_threshold }}</span> &nbsp;|&nbsp;
    Infer every: <span>{{ infer_skip }} frames</span> &nbsp;|&nbsp;
    Cooldown: <span>{{ cooldown }}s</span> &nbsp;|&nbsp;
    Webhook: <span>{{ webhook }}</span>
  </p>
</body>
</html>
"""

from flask import render_template_string

@app.route("/")
def index():
    return render_template_string(
        PAGE,
        face_hef=FACE_HEF_PATH,
        phone_hef=PHONE_HEF_PATH,
        hand_hef=HAND_HEF_PATH,
        conf=DOOMSCROLL_CONF,
        vote_window=VOTE_WINDOW,
        vote_threshold=f"{VOTE_THRESHOLD:.0%}",
        infer_skip=INFER_EVERY_N_FRAMES,
        cooldown=COOLDOWN_SEC,
        webhook=WEBHOOK_URL,
    )


@app.route("/video_feed")
def video_feed():
    from flask import Response
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=capture_and_infer, daemon=True)
    t.start()
    print("Server running → http://0.0.0.0:5002")
    app.run(host="0.0.0.0", port=5002, debug=False)