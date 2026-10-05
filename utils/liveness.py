import cv2
import numpy as np
import time
from collections import deque

class LivenessDetector:
    """
    Enterprise Multi-Factor Anti-Spoofing & Liveness Engine.
    Combines:
    - Per-Face Spatial Tracking & Multi-Frame Stability Verification (requires 5 consecutive live frames)
    - Smartphone Frame & Screen Bezel Contour Detection
    - Sobel 3D Surface Gradient Depth vs 2D Flat Photo Analysis
    - 2D FFT Moire Spectrum Grid Detection
    - Screen Backlight Specular Glare & HSV Color Gamut Verification
    - Dynamic Temporal Micro-Movement Analysis (No buffering free points)
    """
    def __init__(self):
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Track face histories independently:
        # {track_id: {"bbox": [...], "buffer": deque(...), "live_counter": int, "last_seen": timestamp}}
        self.tracked_faces = {}
        self.next_track_id = 1

    def _get_track_id(self, bbox):
        """Matches current face bbox with existing tracks using Intersection over Union (IoU)."""
        x1, y1, x2, y2 = bbox
        area_current = (x2 - x1) * (y2 - y1)

        best_id = None
        best_iou = 0.0
        now = time.time()

        # Purge stale tracks (> 3 seconds missing)
        stale_ids = [tid for tid, data in self.tracked_faces.items() if now - data["last_seen"] > 3.0]
        for tid in stale_ids:
            del self.tracked_faces[tid]

        for tid, data in self.tracked_faces.items():
            bx1, by1, bx2, by2 = data["bbox"]
            area_b = (bx2 - bx1) * (by2 - by1)

            # Intersection
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = area_current + area_b - intersection
                iou = intersection / float(union)

                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

        if best_id is not None and best_iou > 0.35:
            self.tracked_faces[best_id]["bbox"] = bbox
            self.tracked_faces[best_id]["last_seen"] = now
            return best_id
        else:
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracked_faces[tid] = {
                "bbox": bbox,
                "buffer": deque(maxlen=10),
                "live_counter": 0,
                "last_seen": now
            }
            return tid

    def detect_phone_bezel(self, frame, bbox):
        """
        Detects phone body/screen bezels and rectangular screen borders around the face.
        """
        h_img, w_img, _ = frame.shape
        x1, y1, x2, y2 = bbox
        w_face = x2 - x1
        h_face = y2 - y1

        pad_w = int(w_face * 0.70)
        pad_h = int(h_face * 0.70)

        rx1 = max(0, x1 - pad_w)
        ry1 = max(0, y1 - pad_h)
        rx2 = min(w_img, x2 + pad_w)
        ry2 = min(h_img, y2 + pad_h)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0 or roi.shape[0] < 50 or roi.shape[1] < 50:
            return 1.0, False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 130)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        phone_bezel_found = False
        bezel_score = 1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (w_face * h_face * 0.7):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                if len(approx) == 4:
                    _, _, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(h) / max(1, w)
                    if 1.2 <= aspect_ratio <= 2.6:
                        phone_bezel_found = True
                        bezel_score = 0.0
                        break

        return bezel_score, phone_bezel_found

    def analyze_sobel_3d_gradient(self, face_crop):
        """
        Calculates 3D surface gradient magnitude variance using Sobel operators.
        Real 3D human faces have rich 3D shading curvature gradient variance across nose/cheeks.
        2D printed paper photos or flat phone screens exhibit uniform flat gradient distribution.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.5

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        grad_std = np.std(grad_mag)
        grad_mean = np.mean(grad_mag)

        # 2D paper photos have low gradient variance (< 12.0) or flat artificial print edges (> 65.0)
        if grad_std < 11.0:
            return 0.20  # Flat 2D Photo Print
        elif grad_std > 70.0:
            return 0.30  # Digital screen pixel grid noise
        else:
            return 1.0  # Natural 3D facial curvature

    def analyze_texture_fft(self, face_crop):
        """
        Analyzes 2D Discrete Fourier Transform for screen Moire patterns and digital display noise.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.5, "Invalid Crop"

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 40 or w < 40:
            return 0.5, "Crop too small"

        gray_resized = cv2.resize(gray, (128, 128))
        f = np.fft.fft2(gray_resized)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        cy, cx = 64, 64
        r = 14
        y, x = np.ogrid[:128, :128]
        mask = (x - cx)**2 + (y - cy)**2 > r**2

        high_freq = magnitude_spectrum * mask
        high_freq_mean = np.mean(high_freq[mask])
        high_freq_std = np.std(high_freq[mask])
        max_peak = np.max(high_freq[mask])

        peak_ratio = max_peak / (high_freq_mean + 1e-5)

        texture_score = 1.0
        if peak_ratio > 3.3:
            texture_score -= 0.55
        elif peak_ratio > 2.8:
            texture_score -= 0.35

        if high_freq_std > 28.0:
            texture_score -= 0.35

        return max(0.0, min(1.0, texture_score)), f"PeakRatio={peak_ratio:.2f}"

    def analyze_color_space(self, face_crop):
        """
        Analyzes YCrCb skin gamut & HSV backlight glare saturation distortion.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.5

        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        _, Cr, Cb = cv2.split(ycrcb)

        cr_mean = np.mean(Cr)
        cb_mean = np.mean(Cb)

        is_skin_cr = 132 <= cr_mean <= 173
        is_skin_cb = 77 <= cb_mean <= 128

        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        _, S, V = cv2.split(hsv)
        s_mean = np.mean(S)
        s_std = np.std(S)

        # Detect screen backlight glare (high brightness V > 215, low saturation S < 30)
        glare_pixels = np.sum((V > 215) & (S < 30))
        glare_ratio = glare_pixels / float(V.size)

        score = 1.0
        if not (is_skin_cr and is_skin_cb):
            score -= 0.50  # Screen color distortion

        if glare_ratio > 0.03:
            score -= 0.45  # Screen specular reflection

        if s_std < 10.0 or s_mean < 18.0:
            score -= 0.40  # Paper print low saturation

        return max(0.0, min(1.0, score))

    def analyze_dynamic_movement(self, track_id, face_crop):
        """
        Analyzes per-face temporal micro-movements across frame history buffers.
        Crucial: Returns 0.0 during initial buffering phase (0 free points!).
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0, "No history"

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray_norm = cv2.resize(gray, (100, 100))

        buffer = self.tracked_faces[track_id]["buffer"]
        buffer.append(gray_norm)

        # STRICT: Do not grant free points during buffering! Require at least 4 frames of history.
        if len(buffer) < 4:
            return 0.0, "Buffering (Verifying...)"

        diffs = []
        frames = list(buffer)
        for i in range(1, len(frames)):
            abs_diff = cv2.absdiff(frames[i], frames[i-1])
            diffs.append(np.mean(abs_diff))

        mean_diff = float(np.mean(diffs))

        # Static photos on screens or paper have near zero movement (< 0.70)
        if mean_diff < 0.70:
            movement_score = 0.0  # Static Photo Detected!
            reason = "STATIC PHOTO (Zero Motion)"
        elif mean_diff < 1.2:
            movement_score = 0.40
            reason = "Low Motion"
        elif mean_diff > 30.0:
            movement_score = 0.30
            reason = "Unnatural Rapid Motion"
        else:
            movement_score = 1.0
            reason = "Natural Live Motion"

        return movement_score, reason

    def check_eyes(self, face_crop):
        if face_crop is None or face_crop.size == 0 or self.eye_cascade.empty():
            return 0.60

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

        return 0.80 if len(eyes) >= 1 else 0.50

    def evaluate(self, frame, bbox):
        """
        Evaluates multi-factor liveness for face bbox: [x1, y1, x2, y2].
        Requires 5 consecutive live frames before confirming REAL classification.
        Returns: (liveness_score: float, is_real: bool, details: dict)
        """
        h_frame, w_frame, _ = frame.shape
        x1, y1, x2, y2 = bbox

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            return 0.0, False, {"reason": "Face area too small"}

        track_id = self._get_track_id([x1, y1, x2, y2])
        face_crop = frame[y1:y2, x1:x2]

        # 1. Smartphone Screen Bezel & Rectangular Frame Detection
        bezel_score, bezel_found = self.detect_phone_bezel(frame, [x1, y1, x2, y2])

        # 2. Sobel 3D Surface Gradient Depth vs 2D Flat Photo
        sobel_score = self.analyze_sobel_3d_gradient(face_crop)

        # 3. 2D FFT Moire Spectrum Analysis
        texture_score, texture_info = self.analyze_texture_fft(face_crop)

        # 4. Color Gamut & Screen Glare Analysis
        color_score = self.analyze_color_space(face_crop)

        # 5. Dynamic Per-Face Micro-Movement Analysis (0.0 during buffering)
        movement_score, movement_reason = self.analyze_dynamic_movement(track_id, face_crop)

        # 6. Eye Structure Check
        eye_score = self.check_eyes(face_crop)

        # Weighted Aggregation: 25% Bezel, 25% Dynamic Motion, 20% 3D Sobel Gradient, 15% FFT Texture, 10% Color, 5% Eye
        final_liveness_score = (
            (0.25 * bezel_score) +
            (0.25 * movement_score) +
            (0.20 * sobel_score) +
            (0.15 * texture_score) +
            (0.10 * color_score) +
            (0.05 * eye_score)
        )

        track_data = self.tracked_faces[track_id]

        # Strict Gating Rules
        if bezel_found:
            final_liveness_score = min(final_liveness_score, 0.25)
            track_data["live_counter"] = 0
            reason = "VERIFYING: HOLD STILL"
        elif movement_score <= 0.10:
            final_liveness_score = min(final_liveness_score, 0.30)
            track_data["live_counter"] = 0
            reason = "VERIFYING: RE-ALIGN FACE"
        elif sobel_score <= 0.30:
            final_liveness_score = min(final_liveness_score, 0.35)
            track_data["live_counter"] = 0
            reason = "VERIFYING: HOLD STILL"
        elif texture_score < 0.50:
            track_data["live_counter"] = 0
            reason = "VERIFYING: SCANNING"
        elif color_score < 0.50:
            track_data["live_counter"] = 0
            reason = "VERIFYING: SCANNING"
        elif final_liveness_score >= 0.68:
            track_data["live_counter"] += 1
            if track_data["live_counter"] < 5:
                reason = f"VERIFYING ({track_data['live_counter']}/5)"
            else:
                reason = "VERIFIED REAL FACE"
        else:
            track_data["live_counter"] = 0
            reason = "VERIFYING: RE-ALIGN FACE"

        # REAL classification REQUIRES passing 5 consecutive live frames!
        is_real = (final_liveness_score >= 0.68) and (track_data["live_counter"] >= 5) and (not bezel_found)

        return round(final_liveness_score, 2), is_real, {
            "reason": reason,
            "bezel_score": round(bezel_score, 2),
            "movement_score": round(movement_score, 2),
            "sobel_score": round(sobel_score, 2),
            "texture_score": round(texture_score, 2),
            "color_score": round(color_score, 2),
            "eye_score": round(eye_score, 2),
            "live_counter": track_data["live_counter"],
            "texture_info": texture_info
        }

_liveness_detector = None

def get_liveness_detector():
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector()
    return _liveness_detector
