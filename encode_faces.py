import os
import cv2
import joblib
import numpy as np

from insightface.app import FaceAnalysis

from config import DATASET_DIR, ENCODINGS_DIR, MODEL_NAME

_face_app = None

def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )
        _face_app.prepare(ctx_id=0)
    return _face_app

def generate_encodings(progress_callback=None):
    app = get_face_app()
    face_database = {}

    if not os.path.exists(DATASET_DIR):
        print(f"[!] Dataset folder '{DATASET_DIR}' does not exist.")
        return face_database

    student_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    total_folders = len(student_folders)

    print(f"\n[*] Scanning {total_folders} registered student folder(s)...\n")

    for idx, student_folder in enumerate(student_folders):
        folder_path = os.path.join(DATASET_DIR, student_folder)
        embeddings = []

        print(f"Processing {student_folder} ({idx+1}/{total_folders})...")

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                continue

            faces = app.get(image)
            if len(faces) == 0:
                # Fallback: add margin padding to tight face crops
                padded = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[128, 128, 128])
                faces = app.get(padded)

            if len(faces) > 0:
                embeddings.append(faces[0].embedding)

        if len(embeddings) > 0:
            average_embedding = np.mean(embeddings, axis=0)
            parts = student_folder.split("_")
            enrollment = parts[0]
            name = "_".join(parts[1:]) if len(parts) > 1 else parts[0]

            face_database[enrollment] = {
                "name": name,
                "embedding": average_embedding
            }

        if progress_callback and total_folders > 0:
            progress_callback((idx + 1) / total_folders)

    os.makedirs(ENCODINGS_DIR, exist_ok=True)
    encodings_path = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")
    joblib.dump(face_database, encodings_path)

    print("\n[+] Face Encoding Completed Successfully!")
    print(f"Students Encoded : {len(face_database)}")
    return face_database

if __name__ == "__main__":
    generate_encodings()