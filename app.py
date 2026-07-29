import os
import time
import cv2
import pandas as pd
import streamlit as st
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Face Authentication & Attendance",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background-color: #0f172a;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 5px;
    }
    .header-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

from database.database import create_database, get_all_students, reset_database
from register import register_student
from encode_faces import generate_encodings
from recognize import process_frame, load_encodings, get_recognize_app
from attendance import get_attendance_logs, export_attendance_report

# Ensure database tables exist
create_database()

# Sidebar Navigation
st.sidebar.markdown("<h2 style='color:#38bdf8;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Dashboard", "👤 Student Registration", "⚡ Model Encodings", "🎥 Live Attendance", "📊 Attendance Reports"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧹 System Management")
if st.sidebar.button("⚠️ Clear & Reset Database", use_container_width=True):
    reset_database()
    # Clear encodings and dataset
    if os.path.exists("encodings"):
        for f in os.listdir("encodings"):
            try:
                os.remove(os.path.join("encodings", f))
            except Exception:
                pass
    if os.path.exists("dataset"):
        import shutil
        for f in os.listdir("dataset"):
            folder = os.path.join("dataset", f)
            if os.path.isdir(folder):
                try:
                    shutil.rmtree(folder)
                except Exception:
                    pass
    st.sidebar.success("Database, dataset & encodings reset cleanly!")
    st.rerun()

# -------------------------------------------------------------
# 🏠 DASHBOARD
# -------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown("<h1 class='header-title'>AI Face Authentication System</h1>", unsafe_allow_html=True)
    st.markdown("### Real-time Face Verification & Automated Attendance Tracker")
    st.markdown("---")

    students = get_all_students()
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_logs = get_attendance_logs(selected_date=today_date)
    encodings = load_encodings()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(students)}</div>
            <div class="metric-label">Registered Students</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(today_logs)}</div>
            <div class="metric-label">Present Today ({today_date})</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(encodings)}</div>
            <div class="metric-label">Encoded Profiles</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Quick Registered Students Overview")
    if students:
        df_students = pd.DataFrame(students)
        st.dataframe(df_students[["id", "roll_no", "name", "department", "created_at"]], use_container_width=True)
    else:
        st.info("No students registered yet. Go to 'Student Registration' to add students.")

# -------------------------------------------------------------
# 👤 STUDENT REGISTRATION
# -------------------------------------------------------------
elif page == "👤 Student Registration":
    st.markdown("<h1 class='header-title'>Student Registration</h1>", unsafe_allow_html=True)
    st.write("Register a new student by entering details and capturing face samples using the webcam.")
    st.markdown("---")

    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.subheader("Student Information")
        name = st.text_input("Full Name", placeholder="e.g. Alex Smith")
        enrollment = st.text_input("Enrollment / Roll Number", placeholder="e.g. EN2026101")
        branch = st.text_input("Department / Branch", placeholder="e.g. Computer Science")

        start_btn = st.button("📷 Start Webcam & Register", type="primary", use_container_width=True)

    with col_cam:
        st.subheader("Live Camera Stream")
        frame_window = st.image([])
        status_box = st.empty()

    if start_btn:
        if not name or not enrollment or not branch:
            st.error("Please fill in all student details before starting capture.")
        else:
            status_box.info("Opening webcam for capture...")
            student_folder = os.path.join("dataset", f"{enrollment}_{name.replace(' ', '_')}")
            os.makedirs(student_folder, exist_ok=True)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam. Make sure your camera is connected.")
            else:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_detector = cv2.CascadeClassifier(cascade_path)

                image_count = 0
                last_capture = 0
                captured_successfully = False

                progress_bar = st.progress(0)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(150, 150))

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        current_time = time.time()
                        if current_time - last_capture >= 0.4:
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
                                cv2.imwrite(os.path.join(student_folder, f"{image_count:03}.jpg"), face)
                                last_capture = current_time
                                progress_bar.progress(image_count / 20)

                    cv2.putText(frame, f"Captured: {image_count}/20", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels="RGB")

                    if image_count >= 20:
                        captured_successfully = True
                        break

                cap.release()

                if captured_successfully:
                    success, msg = register_student(name, enrollment, branch, capture_callback=lambda f: True)
                    if success:
                        status_box.success(f"🎉 {msg}")
                        st.balloons()
                    else:
                        status_box.error(msg)
                else:
                    status_box.error("Face capture was incomplete.")

# -------------------------------------------------------------
# ⚡ MODEL ENCODINGS
# -------------------------------------------------------------
elif page == "⚡ Model Encodings":
    st.markdown("<h1 class='header-title'>Generate Face Encodings</h1>", unsafe_allow_html=True)
    st.write("Extract deep learning embeddings (InsightFace) from registered student images to update recognition profiles.")
    st.markdown("---")

    if st.button("🚀 Train & Generate Encodings", type="primary"):
        progress_bar = st.progress(0)
        status_box = st.empty()
        status_box.info("Scanning dataset and computing face embeddings...")

        def update_progress(val):
            progress_bar.progress(val)

        encodings = generate_encodings(progress_callback=update_progress)

        if encodings:
            status_box.success(f"✅ Successfully generated encodings for {len(encodings)} student(s)!")
            st.json({k: v["name"] for k, v in encodings.items()})
        else:
            status_box.warning("No face encodings could be generated. Please make sure students are registered with face photos.")

# -------------------------------------------------------------
# 🎥 LIVE ATTENDANCE
# -------------------------------------------------------------
elif page == "🎥 Live Attendance":
    st.markdown("<h1 class='header-title'>Real-Time Face Recognition & Anti-Spoofing</h1>", unsafe_allow_html=True)
    st.write("Start webcam recognition to automatically verify authentic student identities and log attendance in real-time.")
    
    col_badge1, col_badge2, col_badge3 = st.columns(3)
    with col_badge1:
        st.info("⚡ **Optimization**: CPU Accelerated (320px)")
    with col_badge2:
        st.success("🛡️ **Anti-Spoofing**: Active (FFT Moire + EAR)")
    with col_badge3:
        fps_metric = st.empty()
        fps_metric.metric("Stream Speed", "-- FPS")

    st.markdown("---")

    database = load_encodings()
    if not database:
        st.warning("⚠️ No face encodings found! Please register students and click 'Train & Generate Encodings' first.")
    else:
        run_cam = st.checkbox("▶️ Start Live Webcam Verification", value=False)
        col_video, col_logs = st.columns([3, 2])
        
        with col_video:
            frame_window = st.image([])
        with col_logs:
            st.subheader("📋 Real-Time Verification Logs")
            log_window = st.empty()

        if run_cam:
            app = get_recognize_app()
            cap = cv2.VideoCapture(0)
            
            # FPS Calculation parameters
            prev_time = time.time()
            frame_count = 0
            fps = 0.0

            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab webcam frame.")
                    break

                curr_time = time.time()
                frame_count += 1
                if curr_time - prev_time >= 1.0:
                    fps = frame_count / (curr_time - prev_time)
                    fps_metric.metric("Stream Speed", f"{fps:.1f} FPS")
                    prev_time = curr_time
                    frame_count = 0

                frame, recognized = process_frame(frame, database, app)

                if recognized:
                    msgs = [f"🎯 {r['name']} ({r['score']:.2f}) | {r['attendance_msg']}" for r in recognized]
                    log_window.success("\n\n".join(msgs))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB")

            cap.release()


# -------------------------------------------------------------
# 📊 ATTENDANCE REPORTS
# -------------------------------------------------------------
elif page == "📊 Attendance Reports":
    st.markdown("<h1 class='header-title'>Attendance Logs & Reports</h1>", unsafe_allow_html=True)
    st.write("View attendance records, filter by date, and export CSV reports.")
    st.markdown("---")

    col_filter, col_export = st.columns([2, 1])
    with col_filter:
        selected_date = st.date_input("Filter by Date", value=datetime.now().date())

    date_str = selected_date.strftime("%Y-%m-%d")
    logs = get_attendance_logs(selected_date=date_str)

    with col_export:
        st.write("")
        st.write("")
        if logs:
            df = pd.DataFrame(logs)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"attendance_{date_str}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

    if logs:
        st.markdown(f"### Records for {date_str}")
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info(f"No attendance records logged for {date_str}.")
