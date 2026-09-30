import base64
from collections import deque
import os
import threading
import time

import cv2
import gdown
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ---------------------------------------------------------------- paths / app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'resnet50v2_model.keras')

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
CORS(app)

MODEL_FILE_ID = "1UInMiIbaHChmI-KSQ7VRMp_53RZpSDd4"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------------------------------------------------------------- yawn model
model = None
model_loaded = False
yawn_infer = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)

    # Compiled graph: much lower per-call overhead than model.predict()
    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
    def yawn_infer(x):
        return model(x, training=False)

    # Warm-up so the first real request isn't slow
    yawn_infer(tf.zeros([1, 224, 224, 3], tf.float32))
    model_loaded = True
    print("Model loaded and warmed up.")
except Exception as e:
    print(f"Error loading model: {e}")

# ---------------------------------------------------------------- mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_det = mp.solutions.face_detection

# static_image_mode=True: every frame is detected from scratch, so a face that
# left the frame is not "tracked" any more.
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
)
# Cheap detector used as a gate + confidence score
face_detector = mp_face_det.FaceDetection(model_selection=0, min_detection_confidence=0.6)

infer_lock = threading.Lock()  # MediaPipe / model are not thread-safe

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

# ---------------------------------------------------------------- settings
EYE_THRESHOLD = 0.25
YAWNING_DURATION = 3          # seconds of continuous yawning -> drowsy
MIN_FACE_WIDTH_FRAC = 0.12    # face must be at least 12% of frame width
YAWN_GATE_MAR = 0.35          # only run the CNN if the mouth is at least this open
YAWN_PROB_THRESHOLD = 0.5

# sensitivity (1-5) -> seconds of closed eyes before "drowsy"
EYES_CLOSED_BY_SENSITIVITY = {1: 2.5, 2: 2.0, 3: 1.5, 4: 1.2, 5: 1.0}
# (A blink lasts ~0.1-0.4 s, so even the most sensitive level stays well above it.)
MAX_FRAME_GAP = 0.8           # seconds; longer gap = dropped frames -> restart timers
RELATIVE_CLOSED_RATIO = 0.75  # "closed" = EAR below 75% of this person's open-eye EAR
EAR_SMOOTH_FRAMES = 3

EYES_CLOSED_START_TIME = None
YAWNING_START_TIME = None
LAST_FRAME_TIME = None
EAR_SMOOTH = deque(maxlen=EAR_SMOOTH_FRAMES)
EAR_HISTORY = deque(maxlen=150)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ---------------------------------------------------------------- helpers
def decode_image(data_url):
    encoded = data_url.split(',')[1] if ',' in data_url else data_url
    arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def pts(landmarks, idx, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx])


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return float((A + B) / (2.0 * C))


def average_ear(landmarks, w, h):
    left = eye_aspect_ratio(pts(landmarks, LEFT_EYE, w, h))
    right = eye_aspect_ratio(pts(landmarks, RIGHT_EYE, w, h))
    return float((left + right) / 2.0)


def mouth_aspect_ratio(landmarks, w, h):
    p = pts(landmarks, [13, 14, 78, 308], w, h)  # inner lips top/bottom, corners
    return float(np.linalg.norm(p[0] - p[1]) / (np.linalg.norm(p[2] - p[3]) + 1e-6))


def extract_mouth(frame, landmarks, w, h):
    points = pts(landmarks, MOUTH_LANDMARKS, w, h).astype(int)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    margin = 10
    x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
    x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
    mouth = frame[y_min:y_max, x_min:x_max]
    if mouth.size == 0:
        return None
    mouth = cv2.resize(mouth, (224, 224)).astype(np.float32) / 255.0
    return np.expand_dims(mouth, axis=0)


def reset_timers():
    global EYES_CLOSED_START_TIME, YAWNING_START_TIME, LAST_FRAME_TIME
    EYES_CLOSED_START_TIME = None
    YAWNING_START_TIME = None
    LAST_FRAME_TIME = None
    EAR_SMOOTH.clear()


# ---------------------------------------------------------------- routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'API is running',
        'model_loaded': bool(model_loaded),
        'endpoints': {'calibrate': '/api/calibrate', 'detect': '/api/detect'},
    })


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global EYE_THRESHOLD
    payload = request.get_json(silent=True) or {}
    frames = payload.get('frames')
    if not frames:
        return jsonify({'error': 'No frames provided'}), 400

    ear_values = []
    with infer_lock:
        for frame_data in frames:
            try:
                frame = decode_image(frame_data)
                if frame is None:
                    continue
                h, w, _ = frame.shape
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not results.multi_face_landmarks:
                    continue
                ear_values.append(average_ear(results.multi_face_landmarks[0].landmark, w, h))
            except Exception as e:
                print(f"Error processing calibration frame: {e}")

    if len(ear_values) < 3:
        return jsonify({'error': 'Face not found in enough calibration frames'}), 400

    # User had eyes closed: threshold must sit ABOVE the closed-eye EAR.
    closed_ear = float(np.median(ear_values))
    EYE_THRESHOLD = float(np.clip(closed_ear + 0.05, 0.15, 0.30))
    return jsonify({'threshold': EYE_THRESHOLD, 'closed_ear': closed_ear,
                    'frames_used': len(ear_values)})


@app.route('/api/detect', methods=['POST'])
def detect_drowsiness():
    global EYES_CLOSED_START_TIME, YAWNING_START_TIME, LAST_FRAME_TIME

    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500

    payload = request.get_json(silent=True) or {}
    if 'image' not in payload:
        return jsonify({'error': 'No image provided'}), 400

    try:
        frame = decode_image(payload['image'])
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        eye_threshold = float(payload.get('eye_threshold', EYE_THRESHOLD))
        sensitivity = int(payload.get('sensitivity', 3))
        closed_limit = EYES_CLOSED_BY_SENSITIVITY.get(sensitivity, 1.0)
        if payload.get('night_mode'):
            frame = enhance_low_light(frame)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        response = {
            'drowsy': False, 'yawning': False, 'eyes_closed': False,
            'face_detected': False, 'face_confidence': 0.0,
            'eye_threshold': eye_threshold,
        }

        with infer_lock:
            # 1) Cheap face gate
            det = face_detector.process(rgb)
            best = None
            if det.detections:
                best = max(det.detections, key=lambda d: d.score[0])
                if best.location_data.relative_bounding_box.width < MIN_FACE_WIDTH_FRAC:
                    best = None
            if best is None:
                reset_timers()
                return jsonify(response)

            # 2) Landmarks
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                reset_timers()
                return jsonify(response)

            response['face_detected'] = True
            response['face_confidence'] = float(best.score[0] * 100.0)
            lm = results.multi_face_landmarks[0].landmark
            now = time.time()

            # Frames were dropped (slow request / skipped frames): timers are stale
            if LAST_FRAME_TIME is not None and now - LAST_FRAME_TIME > MAX_FRAME_GAP:
                EYES_CLOSED_START_TIME = None
                YAWNING_START_TIME = None
                EAR_SMOOTH.clear()
            LAST_FRAME_TIME = now

            # 3) Eyes: smooth EAR so one noisy frame can't flip the state
            EAR_SMOOTH.append(average_ear(lm, w, h))
            ear = float(np.median(EAR_SMOOTH))
            EAR_HISTORY.append(ear)

            # Use the stricter of (slider/calibrated value) and a per-person value
            # learned from recent open-eye readings. Fixes a threshold that is
            # too high for this user/camera angle.
            effective_threshold = eye_threshold
            if len(EAR_HISTORY) >= 20:
                open_ear = float(np.percentile(EAR_HISTORY, 80))
                effective_threshold = min(eye_threshold, open_ear * RELATIVE_CLOSED_RATIO)

            response['eye_threshold'] = effective_threshold
            response['eye_aspect_ratio'] = ear
            response['eyes_closed'] = ear < effective_threshold
            if response['eyes_closed']:
                if EYES_CLOSED_START_TIME is None:
                    EYES_CLOSED_START_TIME = now
                elif now - EYES_CLOSED_START_TIME >= closed_limit:
                    response['drowsy'] = True
            else:
                EYES_CLOSED_START_TIME = None

            # 4) Yawn: run the CNN only if the mouth is clearly open
            mar = mouth_aspect_ratio(lm, w, h)
            response['mouth_aspect_ratio'] = mar
            is_yawning = False
            if mar >= YAWN_GATE_MAR:
                mouth_img = extract_mouth(frame, lm, w, h)
                if mouth_img is not None:
                    prob = float(yawn_infer(tf.constant(mouth_img)).numpy()[0][0])
                    response['yawn_probability'] = prob
                    is_yawning = prob > YAWN_PROB_THRESHOLD

            if is_yawning:
                if YAWNING_START_TIME is None:
                    YAWNING_START_TIME = now
                elif now - YAWNING_START_TIME >= YAWNING_DURATION:
                    response['drowsy'] = True
            else:
                YAWNING_START_TIME = None
            response['yawning'] = is_yawning

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # debug=True starts the reloader, which loads the model twice.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)