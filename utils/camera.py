import cv2
import mediapipe as mp
import os

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

def capture_faces(student_folder):
    cap = cv2.VideoCapture(0)

    count = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)

        if results.detections:

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)

                # Add padding for InsightFace compatibility
                pad_w = int(bw * 0.25)
                pad_h = int(bh * 0.25)

                y1 = max(0, y - pad_h)
                y2 = min(h, y + bh + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(w, x + bw + pad_w)

                face = frame[y1:y2, x1:x2]

                if face.size != 0:
                    count += 1
                    cv2.imwrite(
                        os.path.join(student_folder, f"{count}.jpg"),
                        face
                    )

                    cv2.putText(frame,
                                f"Captured : {count}/20",
                                (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,255,0),
                                2)

        cv2.imshow("Student Registration", frame)

        if count >= 20:
            break

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()