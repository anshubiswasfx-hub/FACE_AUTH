import os
import time
import cv2
import pandas as pd
import streamlit as st
from datetime import datetime

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="FaceAuth AI Pro - Multi-Theme Enterprise Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database.database import create_database, get_all_students, reset_database, get_spoof_logs
from register import register_student
from encode_faces import generate_encodings
from recognize import process_frame, load_encodings, get_recognize_app
from attendance import get_attendance_logs
from utils.stream import ThreadedWebcam
from config import AUTO_GENERATE_ENCODINGS

# Initialize database
create_database()

# -------------------------------------------------------------
# 🎨 MULTI-THEME ENGINE CONFIGURATION
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 15px 0;">
    <h2 style="background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin: 0;">
        🛡️ FaceAuth AI Pro
    </h2>
    <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;">v2.5 PRO</span>
</div>
""", unsafe_allow_html=True)

theme_choice = st.sidebar.selectbox(
    "🎨 UI Color Theme",
    ["🌌 Cyber Neon (Dark)", "🏎️ Midnight Amber (Stealth)", "🪐 Deep Space Mint", "💎 Nordic Frost Glass (Light)"]
)

# Theme CSS Definitions
if theme_choice == "🌌 Cyber Neon (Dark)":
    theme_css = """
    :root {
        --bg-canvas: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #070a14 70%, #03050a 100%);
        --card-bg: rgba(15, 23, 42, 0.70);
        --card-border: rgba(56, 189, 248, 0.18);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent-grad: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        --accent-glow: rgba(56, 189, 248, 0.4);
        --btn-bg: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%);
        --sidebar-bg: #070a12;
    }
    """
elif theme_choice == "🏎️ Midnight Amber (Stealth)":
    theme_css = """
    :root {
        --bg-canvas: radial-gradient(circle at 50% 0%, #291e09 0%, #0d0a05 70%, #040301 100%);
        --card-bg: rgba(24, 20, 15, 0.75);
        --card-border: rgba(245, 158, 11, 0.22);
        --text-primary: #fffbeb;
        --text-secondary: #d97706;
        --accent-grad: linear-gradient(90deg, #f59e0b 0%, #fbbf24 50%, #ef4444 100%);
        --accent-glow: rgba(245, 158, 11, 0.4);
        --btn-bg: linear-gradient(90deg, #d97706 0%, #b45309 100%);
        --sidebar-bg: #0a0804;
    }
    """
elif theme_choice == "🪐 Deep Space Mint":
    theme_css = """
    :root {
        --bg-canvas: radial-gradient(circle at 50% 0%, #064e3b 0%, #041712 70%, #020c09 100%);
        --card-bg: rgba(6, 30, 24, 0.75);
        --card-border: rgba(52, 211, 153, 0.22);
        --text-primary: #ecfdf5;
        --text-secondary: #6ee7b7;
        --accent-grad: linear-gradient(90deg, #34d399 0%, #10b981 50%, #14b8a6 100%);
        --accent-glow: rgba(52, 211, 153, 0.4);
        --btn-bg: linear-gradient(90deg, #059669 0%, #0d9488 100%);
        --sidebar-bg: #03120e;
    }
    """
else:  # 💎 Nordic Frost Glass (Light)
    theme_css = """
    :root {
        --bg-canvas: linear-gradient(135deg, #eef2ff 0%, #f8fafc 50%, #e2e8f0 100%);
        --card-bg: rgba(255, 255, 255, 0.85);
        --card-border: rgba(37, 99, 235, 0.18);
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --accent-grad: linear-gradient(90deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%);
        --accent-glow: rgba(37, 99, 235, 0.3);
        --btn-bg: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        --sidebar-bg: #f1f5f9;
    }
    """

# Inject Dynamic CSS Styling Engine
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    {theme_css}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .stApp {{
        background: var(--bg-canvas);
        color: var(--text-primary);
    }}

    /* Ultra Glassmorphic Cards */
    .glass-card {{
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 35px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    
    .glass-card:hover {{
        border-color: var(--accent-glow);
        transform: translateY(-3px);
        box-shadow: 0 15px 40px 0 var(--accent-glow);
    }}

    /* Metrics Grid Card */
    .metric-container {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }}
    .metric-container:hover {{
        transform: translateY(-2px);
        border-color: var(--accent-glow);
    }}
    .metric-value {{
        font-size: 2.6rem;
        font-weight: 800;
        background: var(--accent-grad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }}
    .metric-value-alert {{
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f87171 0%, #ef4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }}
    .metric-label {{
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 8px;
    }}

    /* Typography */
    .header-title {{
        background: var(--accent-grad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }}
    .sub-title {{
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 25px;
    }}

    /* Sidebar Navigation Overhaul */
    [data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--card-border);
    }}

    /* Custom Button Styling */
    .stButton>button {{
        background: var(--btn-bg);
        color: #ffffff !important;
        border: none;
        border-radius: 14px;
        padding: 10px 24px;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 20px var(--accent-glow);
        transition: all 0.25s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px var(--accent-glow);
    }}

    /* Download Buttons */
    .stDownloadButton>button {{
        background: var(--btn-bg);
        color: #ffffff !important;
        border-radius: 14px;
        font-weight: 700;
    }}

    /* HUD Video Frame Header */
    .hud-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid var(--card-border);
        border-radius: 14px 14px 0 0;
        padding: 10px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #38bdf8;
    }}
    .live-dot {{
        height: 10px;
        width: 10px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px #10b981;
        margin-right: 6px;
        animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# -------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation Menu",
    ["🏠 Dashboard & Analytics", "👤 Student Registration", "⚡ Model Encodings", "🎥 Live Attendance", "📊 Security & Reports"],
    index=0
)

st.sidebar.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08); margin: 20px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color:var(--text-secondary);'>🎛️ AI Security Controls</h4>", unsafe_allow_html=True)

match_threshold = st.sidebar.slider("Match Similarity Threshold", min_value=0.40, max_value=0.90, value=0.60, step=0.05)
liveness_threshold = st.sidebar.slider("Anti-Spoof Strictness", min_value=0.50, max_value=0.90, value=0.68, step=0.03)

st.sidebar.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08); margin: 20px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color:var(--text-secondary);'>⚙️ System Maintenance</h4>", unsafe_allow_html=True)

if st.sidebar.button("⚠️ Clear & Reset Database", use_container_width=True):
    reset_database()
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
    st.sidebar.success("✅ Database & face encodings reset cleanly!")
    st.rerun()

# -------------------------------------------------------------
# 🏠 DASHBOARD & ANALYTICS PAGE
# -------------------------------------------------------------
if page == "🏠 Dashboard & Analytics":
    st.markdown("<h1 class='header-title'>AI Face Authentication System Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Enterprise Real-Time Face Verification, 3D Anti-Spoof Protection & Attendance Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")

    students = get_all_students()
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_logs = get_attendance_logs(selected_date=today_date)
    spoof_attempts = get_spoof_logs()
    encodings = load_encodings()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(students)}</div>
            <div class="metric-label">Registered Students</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(today_logs)}</div>
            <div class="metric-label">Present Today ({today_date})</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(encodings)}</div>
            <div class="metric-label">Encoded Profiles</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value-alert">{len(spoof_attempts)}</div>
            <div class="metric-label">Blocked Spoof Attacks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Department Distribution")
        if students:
            df_stud = pd.DataFrame(students)
            if "department" in df_stud.columns:
                st.bar_chart(df_stud["department"].value_counts())
        else:
            st.info("No student department data enrolled yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🛡️ Blocked Attack Reasons")
        if spoof_attempts:
            df_sp = pd.DataFrame(spoof_attempts)
            if "reason" in df_sp.columns:
                st.bar_chart(df_sp["reason"].value_counts())
        else:
            st.success("🎉 Zero spoof attack threats logged!")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Enrolled Student Roster")
    if students:
        df_students = pd.DataFrame(students)
        st.dataframe(df_students[["id", "roll_no", "name", "department", "created_at"]], use_container_width=True)
    else:
        st.info("ℹ️ No students registered yet. Go to 'Student Registration' to enroll new students.")

# -------------------------------------------------------------
# 👤 STUDENT REGISTRATION PAGE (AUTOMATIC ENCODINGS)
# -------------------------------------------------------------
elif page == "👤 Student Registration":
    st.markdown("<h1 class='header-title'>Student Registration & Auto-Encoding</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Enroll new students with automated face dataset capture and instant deep embedding generation.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📝 Student Profile Information")
        name = st.text_input("Full Name", placeholder="e.g. Alex Smith")
        enrollment = st.text_input("Enrollment / Roll Number", placeholder="e.g. EN2026101")
        branch = st.text_input("Department / Branch", placeholder="e.g. Computer Science")

        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("📷 Start Capture & Auto-Enroll", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cam:
        st.subheader("📹 Live Dataset Capture Stream")
        frame_window = st.image([])
        status_box = st.empty()

    if start_btn:
        if not name.strip() or not enrollment.strip() or not branch.strip():
            st.error("⚠️ Please fill in all student details before starting capture.")
        else:
            status_box.info("📷 Initializing webcam for face capture...")
            student_folder = os.path.join("dataset", f"{enrollment.strip()}_{name.strip().replace(' ', '_')}")
            os.makedirs(student_folder, exist_ok=True)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check hardware connection.")
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
                    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(140, 140))

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (16, 185, 129), 2)
                        current_time = time.time()
                        if current_time - last_capture >= 0.35:
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
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (16, 185, 129), 2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                    if image_count >= 20:
                        captured_successfully = True
                        break

                cap.release()

                if captured_successfully:
                    success, msg = register_student(name, enrollment, branch, capture_callback=lambda f: True)
                    if success:
                        status_box.success(f"🎉 {msg}")
                        
                        if AUTO_GENERATE_ENCODINGS:
                            st.markdown("---")
                            st.info(f"⚡ **Auto-Generating Face Encodings for {name}...**")
                            encode_progress = st.progress(0)
                            
                            def update_progress(val):
                                encode_progress.progress(val)

                            encodings = generate_encodings(progress_callback=update_progress)
                            load_encodings()
                            st.success(f"🚀 **Model Encodings Auto-Generated Successfully!** Total profiles encoded: {len(encodings)}")
                            st.balloons()
                    else:
                        status_box.error(msg)
                else:
                    status_box.error("❌ Face capture was incomplete.")

# -------------------------------------------------------------
# ⚡ MODEL ENCODINGS PAGE
# -------------------------------------------------------------
elif page == "⚡ Model Encodings":
    st.markdown("<h1 class='header-title'>Generate Face Encodings</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Extract InsightFace deep learning embeddings from registered dataset images to update recognition profiles.</p>", unsafe_allow_html=True)
    st.markdown("---")

    enc = load_encodings()
    st.write(f"Currently active face profile encodings: **{len(enc)}**")

    if st.button("🚀 Re-Train & Update Encodings Manually", type="primary"):
        progress_bar = st.progress(0)
        status_box = st.empty()
        status_box.info("Scanning dataset and computing 512-d face embeddings...")

        def update_progress(val):
            progress_bar.progress(val)

        encodings = generate_encodings(progress_callback=update_progress)
        load_encodings()

        if encodings:
            status_box.success(f"✅ Successfully updated encodings for {len(encodings)} student(s)!")
            st.json({k: v["name"] for k, v in encodings.items()})
        else:
            status_box.warning("⚠️ No face encodings could be generated. Please make sure students are registered with face photos.")

# -------------------------------------------------------------
# 🎥 LIVE ATTENDANCE PAGE (LAG-FREE THREADED STREAM)
# -------------------------------------------------------------
elif page == "🎥 Live Attendance":
    st.markdown("<h1 class='header-title'>Real-Time Live Attendance & Anti-Spoofing</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Zero-lag threaded video verification with 5-frame consecutive stability & 3D Sobel depth analysis.</p>", unsafe_allow_html=True)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.info("⚡ **Engine**: Threaded Async Stream (30 FPS)")
    with col_b2:
        st.success(f"🛡️ **Anti-Spoof**: Active (Strictness={liveness_threshold:.2f})")
    with col_b3:
        fps_metric = st.empty()
        fps_metric.metric("Stream Speed", "-- FPS")

    st.markdown("---")

    database = load_encodings()
    if not database:
        st.warning("⚠️ No face encodings found! Please register students first.")
    else:
        run_cam = st.checkbox("▶️ Start Lag-Free Webcam Verification", value=False)
        col_video, col_logs = st.columns([3, 2])
        
        with col_video:
            st.markdown("""
            <div class="hud-header">
                <div><span class="live-dot"></span> LIVE CAMERA MONITOR</div>
                <div style="color: #64748b;">AI VERIFICATION HUD</div>
            </div>
            """, unsafe_allow_html=True)
            frame_window = st.image([])

        with col_logs:
            st.subheader("📋 Real-Time Authentication Log")
            log_window = st.empty()

        if run_cam:
            app = get_recognize_app()
            stream = ThreadedWebcam(src=0).start()
            time.sleep(0.5)

            prev_time = time.time()
            frame_count = 0

            try:
                while run_cam:
                    grabbed, frame = stream.read()
                    if not grabbed or frame is None:
                        time.sleep(0.01)
                        continue

                    curr_time = time.time()
                    frame_count += 1
                    if curr_time - prev_time >= 1.0:
                        fps = frame_count / (curr_time - prev_time)
                        fps_metric.metric("Stream Speed", f"{fps:.1f} FPS")
                        prev_time = curr_time
                        frame_count = 0

                    frame, recognized = process_frame(frame, database, app, match_thresh=match_threshold)

                    if recognized:
                        msgs = [f"🎯 **{r['name']}** ({r['score']:.2f}) — {r['attendance_msg']}" for r in recognized]
                        log_window.success("\n\n".join(msgs))
                    else:
                        log_window.info("🔍 Monitoring video stream... Present authentic face to mark attendance.")

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                    time.sleep(0.01)
            finally:
                stream.stop()

# -------------------------------------------------------------
# 📊 SECURITY & REPORTS PAGE
# -------------------------------------------------------------
elif page == "📊 Security & Reports":
    st.markdown("<h1 class='header-title'>Security Logs & Attendance Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>View attendance logs, track blocked spoof attack attempts, and export CSV reports.</p>", unsafe_allow_html=True)
    st.markdown("---")

    tab_att, tab_spoof = st.tabs(["📋 Attendance Records", "🛡️ Blocked Spoof Attack Logs"])

    with tab_att:
        col_filter, col_export = st.columns([2, 1])
        with col_filter:
            selected_date = st.date_input("Filter Attendance by Date", value=datetime.now().date())

        date_str = selected_date.strftime("%Y-%m-%d")
        logs = get_attendance_logs(selected_date=date_str)

        with col_export:
            st.write("")
            st.write("")
            if logs:
                df = pd.DataFrame(logs)
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Attendance CSV",
                    data=csv_data,
                    file_name=f"attendance_{date_str}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

        if logs:
            st.markdown(f"### Attendance Logs ({date_str})")
            st.dataframe(pd.DataFrame(logs), use_container_width=True)
        else:
            st.info(f"ℹ️ No attendance records logged for {date_str}.")

    with tab_spoof:
        spoofs = get_spoof_logs(limit=100)
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            st.markdown("### Blocked Spoof Attack Threats")
        with col_s2:
            if spoofs:
                df_sp = pd.DataFrame(spoofs)
                csv_sp = df_sp.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Spoof Logs CSV",
                    data=csv_sp,
                    file_name=f"spoof_threats_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    type="secondary",
                    use_container_width=True
                )

        if spoofs:
            st.dataframe(pd.DataFrame(spoofs), use_container_width=True)
        else:
            st.success("🎉 No spoof attacks detected yet!")
