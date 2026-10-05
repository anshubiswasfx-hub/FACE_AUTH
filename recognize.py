import os
import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis

from config import (
    MODEL_NAME, 
    ENCODINGS_DIR, 
    MATCH_THRESHOLD, 
    DET_SIZE, 
    PROCESSING_SCALE,
    FRAME_SKIP_INTERVAL,
    LIVENESS_ENABLED
)
from attendance import mark_attendance
from database.database import log_spoof_attempt
from utils.liveness import get_liveness_detector

_recognize_app = None
_encodings_matrix = None
_encodings_keys = []
_database_cache = None

# Frame-skipping cache variables for stream acceleration
_frame_counter = 0
_cached_detections = []
_cached_recognized = []
_last_spoof_log_time = 0

def get_recognize_app():
    global _recognize_app
    if _recognize_app is None:
        _recognize_app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )
        _recognize_app.prepare(ctx_id=0, det_size=DET_SIZE)
    return _recognize_app

def load_encodings():
    global _encodings_matrix, _encodings_keys, _database_cache
    encodings_path = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")
    if not os.path.exists(encodings_path):
        _encodings_matrix = None
        _encodings_keys = []
        _database_cache = {}
        return {}
    try:
        data = joblib.load(encodings_path)
        _database_cache = data
        
        if data:
            keys = list(data.keys())
            matrix = np.array([data[k]["embedding"] for k in keys], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            _encodings_matrix = matrix / norms
            _encodings_keys = keys
        else:
            _encodings_matrix = None
            _encodings_keys = []

        return data
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return {}

def draw_sleek_bbox(img, bbox, color, label, subtext):
    x1, y1, x2, y2 = bbox
    h_orig, w_orig = img.shape[:2]

    # Calculate face center and radius axes for circular face ring
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx = int((x2 - x1) * 0.60)
    ry = int((y2 - y1) * 0.68)

    # Base smooth circular ring around face
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)

    # Completing scanning arc circle overlay (300 degree arc)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, -90, 270, color, 4, cv2.LINE_AA)

    # Sleek reticle corners around bbox
    line_len = max(10, int(min(x2 - x1, y2 - y1) * 0.18))
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, 3, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, 3, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, 3, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, 3, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, 3, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, 3, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, 3, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, 3, cv2.LINE_AA)

    # Top Pill Header
    display_label = f"✔ {label}" if "MATCHED" in label else label
    (w_lbl, h_lbl), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    header_y1 = max(0, y1 - h_lbl - 12)
    cv2.rectangle(img, (x1, header_y1), (x1 + w_lbl + 16, y1), color, -1)
    cv2.putText(img, display_label, (x1 + 8, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom Subtext Badge
    if subtext:
        (w_sub, h_sub), _ = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        sub_y1 = min(h_orig - 5, y2 + h_sub + 8)
        cv2.rectangle(img, (x1, y2), (x1 + w_sub + 12, sub_y1), (15, 23, 42), -1)
        cv2.putText(img, subtext, (x1 + 6, sub_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def process_frame(frame, database, app=None, match_thresh=MATCH_THRESHOLD):
    global _encodings_matrix, _encodings_keys, _database_cache
    global _frame_counter, _cached_detections, _cached_recognized, _last_spoof_log_time

    if app is None:
        app = get_recognize_app()

    if frame is None:
        return frame, []

    if _database_cache != database or _encodings_matrix is None:
        load_encodings()

    _frame_counter += 1
    h_orig, w_orig = frame.shape[:2]

    should_detect = (_frame_counter % max(1, FRAME_SKIP_INTERVAL) == 0) or len(_cached_detections) == 0

    if should_detect:
        if PROCESSING_SCALE < 1.0:
            proc_w = int(w_orig * PROCESSING_SCALE)
            proc_h = int(h_orig * PROCESSING_SCALE)
            proc_frame = cv2.resize(frame, (proc_w, proc_h))
            scale_x = w_orig / float(proc_w)
            scale_y = h_orig / float(proc_h)
        else:
            proc_frame = frame
            scale_x = 1.0
            scale_y = 1.0

        faces = app.get(proc_frame)
        liveness_detector = get_liveness_detector() if LIVENESS_ENABLED else None

        new_detections = []
        new_recognized = []
        now = os.times().elapsed if hasattr(os.times(), 'elapsed') else 0

        for face in faces:
            bbox = face.bbox.astype(float)
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)

            embedding = face.embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            norm_emb = embedding / norm if norm > 0 else embedding

            best_name = "Unknown"
            best_enrollment = ""
            best_score = -1.0

            if _encodings_matrix is not None and len(_encodings_keys) > 0:
                sims = np.dot(_encodings_matrix, norm_emb)
                best_idx = np.argmax(sims)
                best_score = float(sims[best_idx])

                if best_score >= match_thresh:
                    best_enrollment = _encodings_keys[best_idx]
                    best_name = database.get(best_enrollment, {}).get("name", "Unknown")

            is_live = True
            liveness_score = 1.0
            liveness_info = {}

            if LIVENESS_ENABLED and liveness_detector:
                liveness_score, is_live, liveness_info = liveness_detector.evaluate(frame, [x1, y1, x2, y2])

            if not is_live:
                color = (0, 215, 255)  # Vibrant Yellow BGR
                label = "VERIFYING..."
                subtext = "Face Verification Active"
            elif best_score >= match_thresh:
                success, msg, student_info = mark_attendance(best_enrollment)
                if success:
                    color = (34, 197, 94)  # Emerald Green
                    label = f"MATCHED: {best_name}"
                    subtext = "Attendance Marked!"
                    is_new = True
                else:
                    color = (50, 160, 235)  # Warm Amber BGR
                    label = f"PRESENT: {best_name}"
                    subtext = "Already Marked Today"
                    is_new = False
                
                new_recognized.append({
                    "enrollment": best_enrollment,
                    "name": best_name,
                    "score": best_score,
                    "attendance_msg": msg,
                    "is_new_mark": is_new
                })
            else:
                color = (225, 29, 72)  # Rose Red
                label = "UNKNOWN"
                subtext = "Unregistered Face"

            new_detections.append({
                "bbox": [x1, y1, x2, y2],
                "color": color,
                "label": label,
                "subtext": subtext
            })

        _cached_detections = new_detections
        _cached_recognized = new_recognized

    # Render bounding boxes and status labels on current frame
    for item in _cached_detections:
        draw_sleek_bbox(frame, item["bbox"], item["color"], item["label"], item["subtext"])

    return frame, _cached_recognized

def run_recognition():
    app = get_recognize_app()
    database = load_encodings()

    if not database:
        print("[!] No face encodings found! Please register students first.")
        return

    print(f"[+] Loaded {len(database)} student encoding(s)")
    cap = cv2.VideoCapture(0)
    print("[*] Webcam Started (Press 'q' to exit)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, recognized = process_frame(frame, database, app)

        for item in recognized:
            print(f"[*] {item['attendance_msg']}")

        cv2.imshow("AI Face Authentication & Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()