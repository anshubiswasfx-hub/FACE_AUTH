import cv2
import numpy as np
import time

class LivenessDetector:
    def __init__(self):
        # OpenCV built-in eye cascade for lightweight eye verification
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def analyze_texture_fft(self, face_crop):
        """
        Detects digital screen Moire patterns and printed photo grain using 2D Fast Fourier Transform.
        Digital screens produce unnatural periodic frequency spikes.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.5, "Invalid Crop"

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 40 or w < 40:
            return 0.5, "Crop too small"

        # Resize to fixed size for consistent frequency analysis
        gray_resized = cv2.resize(gray, (128, 128))
        
        # 2D Discrete Fourier Transform
        f = np.fft.fft2(gray_resized)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # Center coordinates
        cy, cx = 64, 64
        # Mask center low frequencies
        r = 12
        y, x = np.ogrid[:128, :128]
        mask = (x - cx)**2 + (y - cy)**2 > r**2
        
        high_freq = magnitude_spectrum * mask
        high_freq_mean = np.mean(high_freq[mask])
        high_freq_std = np.std(high_freq[mask])
        
        # Screens have high peak-to-mean ratio due to pixel grid interference
        max_peak = np.max(high_freq[mask])
        peak_ratio = max_peak / (high_freq_mean + 1e-5)

        texture_score = 1.0
        if peak_ratio > 3.8:
            texture_score -= 0.45
        elif peak_ratio > 3.2:
            texture_score -= 0.25

        if high_freq_std > 35.0:
            texture_score -= 0.25

        return max(0.0, min(1.0, texture_score)), f"PeakRatio={peak_ratio:.2f}"

    def analyze_color_space(self, face_crop):
        """
        Analyzes skin tone distribution in YCrCb and HSV color space.
        Screens exhibit RGB LED backlight balance differences and paper prints have compressed gamut.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.5

        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # Standard human skin Cr and Cb bounds
        cr_mean = np.mean(Cr)
        cb_mean = np.mean(Cb)

        # Human skin typically has Cr between 133 and 173, Cb between 77 and 127
        is_skin_cr = 133 <= cr_mean <= 173
        is_skin_cb = 77 <= cb_mean <= 127

        # Calculate Color Variance
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        s_std = np.std(S)
        v_std = np.std(V)

        score = 1.0
        if not (is_skin_cr and is_skin_cb):
            score -= 0.35
        
        # Screen glare / reflections cause extreme saturation/value variance
        if s_std < 10.0 or v_std > 85.0:
            score -= 0.25

        return max(0.0, min(1.0, score))

    def analyze_sharpness_variance(self, face_crop):
        """
        Calculates Laplacian variance for depth falloff vs 2D print/screen flatness.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.5
            
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Photos re-captured on phone screens often have extreme blur (< 30) or extreme edge noise (> 800)
        score = 1.0
        if laplacian_var < 35.0:
            score -= 0.40  # Screen blur / photo blur
        elif laplacian_var > 750.0:
            score -= 0.30  # Screen pixel noise / glare

        return max(0.0, min(1.0, score))

    def check_eyes(self, face_crop):
        """
        Verifies presence of eyes using OpenCV cascade inside the face crop.
        """
        if face_crop is None or face_crop.size == 0 or self.eye_cascade.empty():
            return 0.7

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        
        if len(eyes) >= 1:
            return 0.90
        else:
            return 0.60

    def evaluate(self, frame, bbox):
        """
        Evaluates liveness of a detected face bbox: [x1, y1, x2, y2].
        Returns:
            liveness_score (float 0.0 to 1.0),
            is_real (bool),
            details (dict)
        """
        h_frame, w_frame, _ = frame.shape
        x1, y1, x2, y2 = bbox
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            return 0.0, False, {"reason": "Face area too small"}

        face_crop = frame[y1:y2, x1:x2]

        # 1. Texture & FFT Moire Analysis
        texture_score, texture_info = self.analyze_texture_fft(face_crop)

        # 2. Color Gamut Analysis
        color_score = self.analyze_color_space(face_crop)

        # 3. Laplacian Depth Blur
        blur_score = self.analyze_sharpness_variance(face_crop)

        # 4. Eye Check
        eye_score = self.check_eyes(face_crop)

        # Weighted Aggregation
        # 35% Texture/FFT, 30% Color, 20% Blur, 15% Eye check
        final_liveness_score = (0.35 * texture_score) + (0.30 * color_score) + (0.20 * blur_score) + (0.15 * eye_score)

        is_real = final_liveness_score >= 0.55

        reason = "REAL HUMAN" if is_real else "SPOOF / PHOTO DETECTED"

        return round(final_liveness_score, 2), is_real, {
            "reason": reason,
            "texture_score": round(texture_score, 2),
            "color_score": round(color_score, 2),
            "blur_score": round(blur_score, 2),
            "eye_score": round(eye_score, 2),
            "texture_info": texture_info
        }

_liveness_detector = None

def get_liveness_detector():
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector()
    return _liveness_detector
