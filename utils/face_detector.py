import cv2
import os
import time


def capture_faces(student_folder):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return False

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("❌ Failed to load Haar Cascade.")
        print(cascade_path)
        cap.release()
        return False

    image_count = 0
    last_capture = 0

    print("\n📷 Face Capture Started")
    print("Press Q to cancel.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(150, 150)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_capture >= 0.5:
                # Add padding (25% on each side) so InsightFace detection works reliably
                pad_w = int(w * 0.25)
                pad_h = int(h * 0.25)
                h_img, w_img, _ = frame.shape
                
                y1 = max(0, y - pad_h)
                y2 = min(h_img, y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(w_img, x + w + pad_w)

                face = frame[y1:y2, x1:x2]

                if face.size > 0:
                    image_count += 1
                    filename = os.path.join(
                        student_folder,
                        f"{image_count:03}.jpg"
                    )
                    cv2.imwrite(filename, face)
                    last_capture = current_time

        cv2.putText(
            frame,
            f"Captured : {image_count}/20",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Student Registration", frame)

        if image_count >= 20:
            print("\n✅ 20 Images Captured")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n❌ Capture Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

    return image_count == 20