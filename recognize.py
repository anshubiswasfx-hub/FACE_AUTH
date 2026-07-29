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
    LIVENESS_ENABLED,
    LIVENESS_THRESHOLD
)
from attendance import mark_attendance
from utils.liveness import get_liveness_detector

_recognize_app = None
_encodings_matrix = None
_encodings_keys = []
_database_cache = None

def get_recognize_app():
    global _recognize_app
    if _recognize_app is None:
        _recognize_app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )
        # Setting det_size=(320, 320) dramatically speeds up CPU detection by ~4x-8x
        _recognize_app.prepare(ctx_id=0, det_size=DET_SIZE)
    return _recognize_app

def load_encodings():
    global _encodings_matrix, _encodings_keys, _database_cache
    encodings_path = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")
    if not os.path.exists(encodings_path):
        return {}
    try:
        data = joblib.load(encodings_path)
        _database_cache = data
        
        # Build vectorized matrix for fast batch cosine similarity comparison
        if data:
            keys = list(data.keys())
            matrix = np.array([data[k]["embedding"] for k in keys], dtype=np.float32)
            # Pre-normalize matrix vectors for instant dot product cosine similarity
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

def process_frame(frame, database, app=None):
    global _encodings_matrix, _encodings_keys, _database_cache
    if app is None:
        app = get_recognize_app()

    recognized_list = []
    if frame is None:
        return frame, recognized_list

    if _database_cache != database or _encodings_matrix is None:
        load_encodings()

    h_orig, w_orig = frame.shape[:2]

    # Speed Optimization: Downscale frame for fast InsightFace detection if needed
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

    for face in faces:
        # Scale bounding box back to original frame dimensions
        bbox = face.bbox.astype(float)
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)

        # 1. Vectorized Embedding Match against Student Database
        embedding = face.embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            norm_emb = embedding / norm
        else:
            norm_emb = embedding

        best_name = "Unknown"
        best_enrollment = ""
        best_score = -1.0

        if _encodings_matrix is not None and len(_encodings_keys) > 0:
            # Fast vectorized dot product cosine similarity
            sims = np.dot(_encodings_matrix, norm_emb)
            best_idx = np.argmax(sims)
            best_score = float(sims[best_idx])

            if best_score >= MATCH_THRESHOLD:
                best_enrollment = _encodings_keys[best_idx]
                best_name = database[best_enrollment]["name"]

        # 2. Smart Anti-Spoofing & Liveness Check
        is_live = True
        liveness_score = 1.0
        liveness_info = {}

        if LIVENESS_ENABLED and liveness_detector:
            liveness_score, is_live, liveness_info = liveness_detector.evaluate(frame, [x1, y1, x2, y2])

        # 3. Decision Logic & Visualization
        if not is_live:
            # Anti-spoofing triggered (Photo or Video detected!)
            color = (0, 0, 255) # Red
            label = f"SPOOF / PHOTO ({liveness_score:.2f})"
            status_text = "FAKE / PHOTO DETECTED"
        elif best_score >= MATCH_THRESHOLD:
            # Verified Real Human Face & Matched Student
            color = (0, 255, 0) # Green
            label = f"REAL: {best_name} ({best_score:.2f})"
            status_text = "AUTHENTICATED"
            
            # Automatically record attendance only if real human
            success, msg, student_info = mark_attendance(best_enrollment)
            recognized_list.append({
                "enrollment": best_enrollment,
                "name": best_name,
                "score": best_score,
                "liveness_score": liveness_score,
                "attendance_msg": msg
            })
        else:
            # Verified Real Human Face, but Unknown identity
            color = (0, 165, 255) # Orange
            label = f"REAL: Unknown ({max(0, best_score):.2f})"
            status_text = "UNKNOWN FACE"

        # Draw polished overlay
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Top banner label
        cv2.rectangle(frame, (x1, max(0, y1 - 30)), (x2, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 6, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Bottom badge
        cv2.putText(
            frame,
            f"Liveness: {liveness_score:.2f}",
            (x1, min(h_orig - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame, recognized_list

def run_recognition():
    app = get_recognize_app()
    database = load_encodings()

    if not database:
        print("❌ No face encodings found! Please run 'python encode_faces.py' first.")
        return

    print(f"✅ Loaded {len(database)} student encoding(s)")
    cap = cv2.VideoCapture(0)
    print("📷 Webcam Started (Press 'q' to exit)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, recognized = process_frame(frame, database, app)

        for item in recognized:
            print(f"🎯 {item['attendance_msg']}")

        cv2.imshow("AI Face Authentication & Attendance (Anti-Spoofing Protected)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()