# Folder Paths
DATASET_DIR = "dataset"
ENCODINGS_DIR = "encodings"
DATABASE_PATH = "database/students.db"
REPORTS_DIR = "reports"

# Camera & Streaming
CAMERA_ID = 0
FRAME_SKIP_INTERVAL = 2  # Run deep model every 2nd frame for smooth 30 FPS stream

# Recognition & AI Model
MATCH_THRESHOLD = 0.60
MODEL_NAME = "buffalo_l"
DET_SIZE = (320, 320)
PROCESSING_SCALE = 0.5

# Anti-Spoofing & Liveness Settings
LIVENESS_ENABLED = True
LIVENESS_THRESHOLD = 0.65
AUTO_GENERATE_ENCODINGS = True

# Advanced Challenge-Response Liveness Mode
EAR_BLINK_THRESHOLD = 0.20
CHALLENGE_MODE_ENABLED = False