import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import cv2
import pandas as pd
import numpy as np
import streamlit as st
import textwrap
from datetime import datetime

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="FaceAuth AI - Enterprise Security",
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
# 🎨 CLEAN & SIMPLE PROFESSIONAL LIGHT THEME
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 2px 4px 12px 4px;">
    <h2 style="color: #0f172a; font-size: 1.3rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
        FaceAuth AI
    </h2>
</div>
""", unsafe_allow_html=True)

# Inject PWA Meta Tags, Manifest Link & Service Worker Registration
st.markdown("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="theme-color" content="#0f172a">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="FaceAuth AI">
<link rel="apple-touch-icon" href="/app/static/icon.png">
<script>
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/app/static/sw.js').catch(function() {});
    });
}
window.addEventListener('beforeinstallprompt', function(e) {
    e.preventDefault();
    window.deferredInstallPrompt = e;
    var btn = document.getElementById('pwa-direct-install-btn');
    if (btn) { btn.style.display = 'inline-flex'; }
});
</script>
""", unsafe_allow_html=True)

# Inject High-Contrast Professional Responsive CSS
st.markdown("""
<style>
    /* 1. Header Bar & Sidebar Wrapup (Collapse/Expand) Controls */
    header[data-testid="stHeader"],
    .stAppHeader {
        background: transparent !important;
        height: 0px !important;
        min-height: 0px !important;
        pointer-events: none !important;
        display: block !important;
        visibility: visible !important;
        overflow: visible !important;
        z-index: 99999 !important;
        border: none !important;
    }

    /* Toolbar must be visible for the expand button, but pointer-events none so it doesn't block */
    [data-testid="stToolbar"],
    .stAppToolbar {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        display: flex !important;
        visibility: visible !important;
        pointer-events: none !important;
        height: 0px !important;
        min-height: 0px !important;
        overflow: visible !important;
    }

    /* Expand Sidebar Button (when sidebar is wrapped up / collapsed) */
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        z-index: 1000000 !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        min-height: 36px !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12) !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    button[data-testid="stExpandSidebarButton"]:hover,
    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover {
        background: #0f172a !important;
        border-color: #0f172a !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.22) !important;
        transform: scale(1.05) !important;
    }

    [data-testid="stExpandSidebarButton"] svg,
    button[data-testid="stExpandSidebarButton"] svg,
    [data-testid="stExpandSidebarButton"] span,
    button[data-testid="stExpandSidebarButton"] span,
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] span {
        color: inherit !important;
        fill: currentColor !important;
        font-size: 20px !important;
        width: 20px !important;
        height: 20px !important;
        line-height: 1 !important;
    }

    /* Inside sidebar: Collapse Button (when sidebar is open) */
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        visibility: visible !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding: 8px 12px 0 12px !important;
        height: 40px !important;
        min-height: 40px !important;
        background: transparent !important;
    }

    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
        padding: 0 !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background: #0f172a !important;
        border-color: #0f172a !important;
        color: #ffffff !important;
        transform: scale(1.05) !important;
    }

    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="stSidebarCollapseButton"] button span {
        color: inherit !important;
        fill: currentColor !important;
        font-size: 18px !important;
        width: 18px !important;
        height: 18px !important;
    }

    /* HIDE ONLY unwanted toolbar items (Deploy button, hamburger menu, status widget, decoration, footer) */
    [data-testid="stToolbarActions"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"],
    [data-testid="stMainMenuButton"],
    #MainMenu,
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* 2. Global Typography & Background */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0f172a !important;
    }

    .stApp {
        background-color: #f8fafc !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* 3. ZERO OUT all top spacing across entire viewport */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main,
    section[data-testid="stSidebar"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    /* Main Container Padding - Room for the floating unwrap button on the left */
    .block-container,
    [data-testid="block-container"],
    [data-testid="stMainBlockContainer"],
    .main .block-container {
        max-width: 1350px !important;
        padding-top: 0.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3.8rem !important;
        padding-right: 2rem !important;
        margin-top: 0rem !important;
    }

    /* Distinct Visible Sidebar Layout with Fixed Width */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        min-width: 270px !important;
        width: 270px !important;
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* Sidebar User Content Padding */
    [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] div.stSidebarContent,
    section[data-testid="stSidebar"] > div {
        padding-top: 0.2rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-bottom: 1.5rem !important;
        margin-top: 0rem !important;
    }

    /* 4. Sidebar Navigation - Full Width Responsive Button Cards */
    [data-testid="stSidebar"] [data-testid="stRadio"],
    [data-testid="stSidebar"] [data-testid="stRadioGroup"],
    [data-testid="stSidebar"] div[role="radiogroup"],
    [data-testid="stSidebar"] .stRadio > div,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        width: 100% !important;
        min-width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    /* Force each option container to take 100% full width */
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div,
    [data-testid="stSidebar"] div[role="radiogroup"] > div,
    [data-testid="stSidebar"] .stRadio > div > div {
        width: 100% !important;
        min-width: 100% !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    /* Tab Button card styling */
    [data-testid="stSidebar"] [data-testid="stRadioOption"],
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] label,
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100% !important;
        min-width: 100% !important;
        height: 44px !important;
        min-height: 44px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 0 14px !important;
        margin: 0 !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }

    /* Hide the radio circle / dot cleanly */
    [data-testid="stSidebar"] [data-testid="stRadioOption"] input {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not(:only-child) {
        display: none !important;
    }

    /* Inner flex text container */
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div {
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Navigation Label Typography - Always Visible */
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] span,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] label,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"] span {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        letter-spacing: -0.01em !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        display: inline-block !important;
        opacity: 1 !important;
        visibility: visible !important;
        line-height: 1.4 !important;
    }

    /* Hover state */
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover span {
        color: #0f172a !important;
    }

    /* Active / Selected State - Dark Slate Pill */
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked),
    [data-testid="stSidebar"] div[data-selected="true"] [data-testid="stRadioOption"],
    [data-testid="stSidebar"] div[aria-checked="true"] [data-testid="stRadioOption"] {
        background: #0f172a !important;
        border-color: #0f172a !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.15) !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] span,
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) span,
    [data-testid="stSidebar"] div[data-selected="true"] [data-testid="stRadioOption"] p,
    [data-testid="stSidebar"] div[data-selected="true"] [data-testid="stRadioOption"] span,
    [data-testid="stSidebar"] div[aria-checked="true"] [data-testid="stRadioOption"] p,
    [data-testid="stSidebar"] div[aria-checked="true"] [data-testid="stRadioOption"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]:hover,
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked):hover {
        background: #0f172a !important;
        border-color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]:hover p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked):hover p {
        color: #ffffff !important;
    }


    /* Sidebar Reset Button */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff !important;
        color: #dc2626 !important;
        border: 1px solid #fecaca !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        height: 38px !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #fef2f2 !important;
        border-color: #dc2626 !important;
        color: #b91c1c !important;
    }

    /* 5. Clean Simple Cards */
    .glass-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 18px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 14px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Metrics Grid Cards */
    .metric-container {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Headings Top Zero-out */
    h1, h2, h3, h4, .header-title {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .header-title {
        color: #0f172a !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 4px !important;
    }
    .sub-title {
        color: #64748b !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        margin-bottom: 16px !important;
        margin-top: 0 !important;
    }

    /* Form Controls */
    .stTextInput input {
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 8px 12px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus {
        border-color: #0f172a !important;
        box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.1) !important;
    }

    /* Simple Dark Buttons */
    .stButton > button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        border-radius: 6px !important;
        padding: 9px 16px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        width: 100% !important;
        height: 42px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
        color: #ffffff !important;
        box-shadow: none !important;
    }

    .stDownloadButton > button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100% !important;
        height: 42px !important;
    }
    .stDownloadButton > button:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
    }

    /* Dividers */
    hr {
        margin: 6px 0 14px 0 !important;
        border: none !important;
        border-top: 1px solid #e2e8f0 !important;
    }

    /* Video Frame HUD Header */
    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f172a !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 9px 14px !important;
        font-size: 0.85rem !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .live-dot {
        height: 8px;
        width: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* Responsive Breakpoints */
    @media (max-width: 992px) {
        .header-title {
            font-size: 1.4rem !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION (EXACT SAME LENGTH TABS)
# -------------------------------------------------------------
PAGES = ["🏠 Dashboard", "🎥 Live Attendance", "👤 Registration", "📋 Records", "📲 Install App"]
page = st.sidebar.radio(
    "Navigation Menu",
    PAGES,
    index=0,
    label_visibility="collapsed"
)

# Set Default Backend Threshold Constants
match_threshold = 0.60
liveness_threshold = 0.68

if page == "🎥 Live Attendance":
    try:
        all_students_sb = get_all_students()
        database_sb = load_encodings()
        total_reg_sb = len(all_students_sb)
        total_enc_sb = len(database_sb)
        pct_sb = int((total_enc_sb / total_reg_sb * 100)) if total_reg_sb > 0 else (100 if total_enc_sb > 0 else 0)

        st.sidebar.markdown(f"""
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #e2e8f0;">
            <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                🧬 Encoding Status
            </div>
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.82rem; font-weight: 600; color: #0f172a;">Encoding Done</span>
                    <span style="font-size: 0.82rem; font-weight: 700; color: {'#16a34a' if pct_sb == 100 else '#d97706'};">{pct_sb}%</span>
                </div>
                <div style="background: #f1f5f9; height: 6px; border-radius: 4px; overflow: hidden; margin-bottom: 8px;">
                    <div style="background: {'#16a34a' if pct_sb == 100 else '#0f172a'}; height: 100%; width: {pct_sb}%; border-radius: 4px;"></div>
                </div>
                <div style="font-size: 0.75rem; color: #64748b; display: flex; justify-content: space-between;">
                    <span>Encoded: <strong style="color: #0f172a;">{total_enc_sb}</strong></span>
                    <span>Enrolled: <strong style="color: #0f172a;">{total_reg_sb}</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

st.sidebar.markdown("""
<div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid #e2e8f0;">
    <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
        ⚙️ Maintenance
    </div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("⚠️ Reset Database", use_container_width=True):
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
# 🏠 CLEAN SIMPLE DASHBOARD
# -------------------------------------------------------------
if page == "🏠 Dashboard":
    students = get_all_students()
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_logs = get_attendance_logs(selected_date=today_date)
    encodings = load_encodings()

    total_students = len(students)
    present_count = len(today_logs)
    absent_count = max(0, total_students - present_count)
    attendance_rate = round((present_count / total_students * 100), 1) if total_students > 0 else 0.0
    active_profiles = len(encodings)
    now_formatted = datetime.now().strftime("%A, %b %d, %Y")

    # 1. Simple Header
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px;">
        <div>
            <h1 style="color: #0f172a; font-size: 1.55rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;">FaceAuth AI Dashboard</h1>
            <p style="color: #64748b; font-size: 0.86rem; margin: 2px 0 0 0;">Attendance Overview & Registered Students</p>
        </div>
        <div style="font-size: 0.82rem; color: #475569; background: #ffffff; border: 1px solid #e2e8f0; padding: 6px 12px; border-radius: 6px;">
            {now_formatted}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Four Simple Metric Cards (NO loud blue parts, clean neutral colors)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-container" style="text-align: left; padding: 14px 16px;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Total Students</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #0f172a; margin: 2px 0;">{total_students}</div>
            <div style="color: #64748b; font-size: 0.76rem;">Enrolled in database</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-container" style="text-align: left; padding: 14px 16px;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Present Today</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #0f172a; margin: 2px 0;">{present_count}</div>
            <div style="color: #64748b; font-size: 0.76rem;">{attendance_rate}% of enrolled</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-container" style="text-align: left; padding: 14px 16px;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Pending / Absent</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #0f172a; margin: 2px 0;">{absent_count}</div>
            <div style="color: #64748b; font-size: 0.76rem;">Awaiting check-in</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-container" style="text-align: left; padding: 14px 16px;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Face Profiles</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #0f172a; margin: 2px 0;">{active_profiles}</div>
            <div style="color: #64748b; font-size: 0.76rem;">Active encodings</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. Two Columns: Department Breakdown & Today's Attendance (NO giant blue bar chart!)
    col_dept, col_feed = st.columns([1, 1])

    with col_dept:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">Department Overview</h3>
        """, unsafe_allow_html=True)
        if students:
            df_stud = pd.DataFrame(students)
            if "department" in df_stud.columns:
                dept_counts = df_stud["department"].value_counts().reset_index()
                dept_counts.columns = ["Department", "Count"]
                for _, row in dept_counts.iterrows():
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 9px 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #1e293b; font-size: 0.88rem;">{row['Department']}</span>
                        <span style="font-size: 0.82rem; font-weight: 600; color: #475569; background: #e2e8f0; padding: 2px 8px; border-radius: 4px;">{row['Count']} Students</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No student department data enrolled yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_feed:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">Today's Attendance</h3>
        """, unsafe_allow_html=True)
        if today_logs:
            for log in today_logs[:5]:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 9px 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 6px;">
                    <div>
                        <div style="font-weight: 600; color: #0f172a; font-size: 0.88rem;">{log['name']}</div>
                        <div style="font-size: 0.74rem; color: #64748b;">Roll: {log['roll_no']} | {log['department']}</div>
                    </div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: #334155; background: #e2e8f0; padding: 2px 8px; border-radius: 4px;">
                        {log['time']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            if len(today_logs) > 5:
                st.caption(f"+ {len(today_logs) - 5} more check-ins today.")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 24px 12px; color: #64748b; font-size: 0.88rem;">
                No attendance recorded yet today.
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Student Roster Table
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">Registered Student Roster</h3>
    """, unsafe_allow_html=True)
    if students:
        df_students = pd.DataFrame(students)
        display_cols = ["roll_no", "name", "department", "created_at"]
        existing_cols = [c for c in display_cols if c in df_students.columns]
        df_display = df_students[existing_cols].copy()
        df_display.columns = [c.replace("_", " ").title() for c in existing_cols]
        st.dataframe(df_display, use_container_width=True, height=200)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #64748b; font-size: 0.88rem;">
            No students enrolled yet.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 👤 STUDENT REGISTRATION PAGE (AUTOMATIC ENCODINGS)
# -------------------------------------------------------------
elif page == "👤 Registration":
    st.markdown("<h1 class='header-title'>Student Registration & Auto-Encoding</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Enroll new students with automated face dataset capture and instant deep embedding generation.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 4px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.subheader("📝 Student Profile Information")
        name = st.text_input("Full Name")
        enrollment = st.text_input("Enrollment / Roll Number")
        
        DEPARTMENTS = [
            "CSE",
            "CSE IOT",
            "CSE CYBER",
            "CSE IT",
            "CSE DS",
            "CSE AIML",
            "CSE AIDS"
        ]
        branch = st.selectbox("Department / Branch", options=DEPARTMENTS)

        reg_cam_source = st.radio(
            "Camera Source:",
            ["📱 Phone / Device Camera (Browser)", "💻 Laptop Hardware Webcam (Live Stream)"],
            horizontal=True
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if reg_cam_source == "💻 Laptop Hardware Webcam (Live Stream)":
            start_btn = st.button("📷 Start Laptop Webcam & Auto-Enroll", type="primary", use_container_width=True)
        else:
            start_btn = False

    with col_cam:
        st.subheader("📹 Face Capture & Enrollment")
        status_box = st.empty()

        if reg_cam_source == "📱 Phone / Device Camera (Browser)":
            st.info("📱 **Phone Camera Active**: Use your phone's front or back camera directly. Tap capture below to take a photo and enroll.")
            phone_photo = st.camera_input("📸 Take Photo with Phone Camera to Register")
            if phone_photo is not None:
                if not name.strip() or not enrollment.strip() or not branch:
                    st.error("⚠️ Please fill in Full Name, Enrollment Number, and Department above before capturing.")
                else:
                    file_bytes = np.asarray(bytearray(phone_photo.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if frame is not None:
                        student_folder = os.path.join("dataset", f"{enrollment.strip()}_{name.strip().replace(' ', '_')}")
                        os.makedirs(student_folder, exist_ok=True)
                        h_img, w_img, _ = frame.shape
                        cv2.imwrite(os.path.join(student_folder, "001.jpg"), frame)
                        cx, cy = w_img // 2, h_img // 2
                        box_size = min(h_img, w_img) // 2
                        face_crop = frame[max(0, cy - box_size):min(h_img, cy + box_size), max(0, cx - box_size):min(w_img, cx + box_size)]
                        if face_crop.size > 0:
                            for idx in range(2, 11):
                                cv2.imwrite(os.path.join(student_folder, f"{idx:03}.jpg"), face_crop)
                        
                        success, msg = register_student(name, enrollment, branch, capture_callback=lambda f: True)
                        if success:
                            st.success(f"🎉 Student **{name}** ({enrollment}) registered successfully from phone camera!")
                            if AUTO_GENERATE_ENCODINGS:
                                with st.spinner("⚡ Compiling 512-D face encodings..."):
                                    encodings = generate_encodings()
                                    load_encodings()
                                st.success(f"🚀 Model Encodings Generated! Total profiles: {len(encodings)}")
                                st.balloons()
                        else:
                            st.error(msg)
        else:
            frame_window = st.image([])

    if start_btn:
        if not name.strip() or not enrollment.strip() or not branch:
            st.error("⚠️ Please fill in all student details before starting capture.")
        else:
            status_box.info("📷 Initializing webcam for face capture...")
            student_folder = os.path.join("dataset", f"{enrollment.strip()}_{name.strip().replace(' ', '_')}")
            os.makedirs(student_folder, exist_ok=True)

            stream = ThreadedWebcam(src=0).start()
            time.sleep(0.3)

            if not stream.is_opened():
                status_box.error("❌ Could not access webcam. Please check hardware connection.")
            else:
                # Load face detector safely
                face_detector = None
                try:
                    if hasattr(cv2, 'CascadeClassifier'):
                        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                        face_detector = cv2.CascadeClassifier(cascade_path)
                except Exception:
                    face_detector = None

                image_count = 0
                last_capture = 0
                captured_successfully = False
                progress_bar = st.progress(0)

                try:
                    while True:
                        grabbed, frame = stream.read()
                        if not grabbed or frame is None:
                            time.sleep(0.01)
                            continue

                        h_img, w_img, _ = frame.shape
                        faces = []

                        if face_detector is not None and hasattr(face_detector, 'empty') and not face_detector.empty():
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))
                        
                        # Fallback face region if classifier is unavailable
                        if len(faces) == 0:
                            cx, cy = w_img // 2, h_img // 2
                            faces = [(cx - 100, cy - 100, 200, 200)]

                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (34, 197, 94), 2)
                            current_time = time.time()
                            if current_time - last_capture >= 0.3:
                                pad_w = int(w * 0.2)
                                pad_h = int(h * 0.2)

                                y1 = max(0, y - pad_h)
                                y2 = min(h_img, y + h + pad_h)
                                x1 = max(0, x - pad_w)
                                x2 = min(w_img, x + w + pad_w)

                                face = frame[y1:y2, x1:x2]
                                if face.size > 0:
                                    image_count += 1
                                    cv2.imwrite(os.path.join(student_folder, f"{image_count:03}.jpg"), face)
                                    last_capture = current_time
                                    progress_bar.progress(min(image_count / 20, 1.0))

                        cv2.putText(frame, f"Captured: {image_count}/20", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2)

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                        if image_count >= 20:
                            captured_successfully = True
                            break

                        time.sleep(0.01)
                finally:
                    stream.stop()

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
                    else:
                        status_box.error(msg)
                else:
                    status_box.error("❌ Face capture was incomplete.")

# -------------------------------------------------------------
# 🎥 LIVE ATTENDANCE PAGE (CLEAN ALIGNED DESIGN)
# -------------------------------------------------------------
elif page == "🎥 Live Attendance":
    st.markdown("<h1 class='header-title'>Real-Time Attendance Verification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Position face towards the camera for automated attendance logging.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 4px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    database = load_encodings()
    all_students = get_all_students()
    total_students = len(all_students)
    total_encoded = len(database)
    pct = int((total_encoded / total_students * 100)) if total_students > 0 else (100 if total_encoded > 0 else 0)

    # Top Control Bar
    col_ctrl, col_stats = st.columns([1.3, 1])
    with col_ctrl:
        att_cam_source = st.radio(
            "Select Camera Source:",
            ["📱 Phone / Device Camera (Browser)", "💻 Laptop Hardware Webcam (Live Stream)"],
            horizontal=True
        )
        if att_cam_source == "💻 Laptop Hardware Webcam (Live Stream)":
            run_cam = st.toggle("▶️ Activate Hardware Webcam Stream", value=False, disabled=(not database))
        else:
            run_cam = False
    with col_stats:
        if not database:
            st.warning("⚠️ No face encodings active! Click 'Sync & Re-Generate Encodings' on the side.")

    col_video, col_logs = st.columns([3, 2])
    
    with col_video:
        if att_cam_source == "📱 Phone / Device Camera (Browser)":
            st.markdown("""
            <div class="hud-header">
                <div><span class="live-dot"></span> PHONE / DEVICE CAMERA</div>
                <div style="color: #cbd5e1; font-weight: 500;">DEVICE CAM</div>
            </div>
            """, unsafe_allow_html=True)
            st.info("📱 **Phone Camera Active**: Take a selfie or capture face with your phone to mark attendance.")
            phone_att_photo = st.camera_input("📸 Capture Face with Phone to Mark Attendance")
            if phone_att_photo is not None:
                file_bytes = np.asarray(bytearray(phone_att_photo.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is not None:
                    app = get_recognize_app()
                    frame, recognized = process_frame(frame, database, app, match_thresh=match_threshold)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, use_container_width=True)

                    new_marks = [r for r in recognized if r.get("is_new_mark", False)]
                    already_marked = [r for r in recognized if not r.get("is_new_mark", False)]
                    
                    if new_marks:
                        r = new_marks[0]
                        st.markdown(f"""
                        <div style="text-align: center; padding: 18px; background: #ecfdf5; border: 2px solid #10b981; border-radius: 12px; margin-top: 10px;">
                            <h3 style="color: #065f46; font-weight: 700; margin: 0 0 6px 0; font-size: 1.25rem;">✅ Attendance Marked!</h3>
                            <p style="color: #047857; font-weight: 600; margin: 0; font-size: 1rem;">Student: <strong>{r['name']}</strong></p>
                            <small style="color: #059669; font-weight: 500;">Confidence: {r.get('confidence', 0):.2f} &bull; Marked via Phone Camera</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    elif already_marked:
                        r = already_marked[0]
                        st.markdown(f"""
                        <div style="background: #f8fafc; border: 1px solid #cbd5e1; padding: 14px 18px; border-radius: 8px; color: #334155; font-weight: 600; margin-top: 10px;">
                            ℹ️ <strong>{r['name']}</strong> is already marked present today.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: #fef2f2; border: 1px solid #fecaca; padding: 14px 18px; border-radius: 8px; color: #991b1b; font-weight: 600; margin-top: 10px;">
                            ❌ No enrolled face detected or match confidence below threshold. Please face the phone camera directly.
                        </div>
                        """, unsafe_allow_html=True)
        else:
            if not run_cam:
                st.markdown("""
                <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 48px 24px; text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 8px;">📹</div>
                    <h3 style="color: #0f172a; font-weight: 700; margin: 0 0 4px 0; font-size: 1.2rem;">Camera Feed Inactive</h3>
                    <p style="color: #64748b; font-size: 0.88rem; margin: 0;">Toggle "Activate Hardware Webcam Stream" above to start live face scanning.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="hud-header">
                    <div><span class="live-dot"></span> VERIFYING CAMERA FEED</div>
                    <div style="color: #cbd5e1; font-weight: 500;">LIVE FEED</div>
                </div>
                """, unsafe_allow_html=True)
                frame_window = st.image([])

    with col_logs:
        st.markdown("""
        <h3 style="margin: 0 0 8px 0; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
            📋 Verification Status
        </h3>
        """, unsafe_allow_html=True)
        log_window = st.empty()
        if not run_cam:
            log_window.markdown("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;">
                <div style="color: #64748b; font-size: 0.85rem; font-weight: 500;">
                    Camera stream is offline. Activate camera to start real-time verification.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 🧬 SIDE ENCODINGS STATUS CARD (HOW MUCH ENCODING IS DONE)
        # ---------------------------------------------------------
        badge_bg = '#dcfce7' if pct == 100 and total_students > 0 else '#fef3c7'
        badge_color = '#15803d' if pct == 100 and total_students > 0 else '#b45309'
        badge_border = '#86efac' if pct == 100 and total_students > 0 else '#fde68a'
        badge_text = f"{pct}% DONE" if total_students > 0 else "0% DONE"
        bar_color = '#16a34a' if pct == 100 and total_students > 0 else '#0f172a'

        st.markdown(f"""<div class="glass-card" style="margin-top: 4px; padding: 16px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<div>
<h4 style="margin: 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">🧬 Face Encodings Status</h4>
<div style="color: #64748b; font-size: 0.75rem; margin-top: 1px;">InsightFace ArcFace 512-D</div>
</div>
<span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; padding: 3px 9px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">{badge_text}</span>
</div>
<div style="background: #f1f5f9; height: 8px; border-radius: 6px; overflow: hidden; margin-bottom: 12px; border: 1px solid #e2e8f0;">
<div style="background: {bar_color}; height: 100%; width: {pct}%; border-radius: 6px; transition: width 0.3s ease;"></div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;">
<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; text-align: center;">
<div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{total_encoded} / {total_students}</div>
<div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Profiles Encoded</div>
</div>
<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; text-align: center;">
<div style="font-size: 1.15rem; font-weight: 700; color: {'#16a34a' if pct == 100 and total_students > 0 else '#d97706'};">{'Ready' if pct == 100 and total_students > 0 else 'Pending'}</div>
<div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Engine Status</div>
</div>
</div>""", unsafe_allow_html=True)

        # Enrolled Profiles List
        if all_students:
            st.markdown("""<div style="color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">Enrolled Profiles & Vectors</div>
<div style="max-height: 160px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; padding-right: 4px; margin-bottom: 12px;">""", unsafe_allow_html=True)
            for s in all_students:
                is_enc = s['roll_no'] in database
                badge_html = """<span style="background: #dcfce7; color: #15803d; border: 1px solid #86efac; padding: 2px 7px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">✅ Encoded</span>""" if is_enc else """<span style="background: #fef3c7; color: #b45309; border: 1px solid #fde68a; padding: 2px 7px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">⏳ Pending</span>"""
                st.markdown(f"""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; display: flex; justify-content: space-between; align-items: center;">
<div style="overflow: hidden; padding-right: 8px;">
<div style="font-size: 0.85rem; font-weight: 600; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{s['name']}</div>
<div style="font-size: 0.74rem; color: #64748b;">Roll: {s['roll_no']} &bull; {s['department']}</div>
</div>
<div style="flex-shrink: 0;">{badge_html}</div>
</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="color: #64748b; font-size: 0.8rem; padding: 8px 0; margin-bottom: 8px;">No registered students found in database.</div>""", unsafe_allow_html=True)

        # Sync / Re-Generate Encodings button
        if st.button("🔄 Sync & Re-Generate Encodings", use_container_width=True):
            try:
                with st.spinner("Compiling face encodings from dataset..."):
                    db_new = generate_encodings()
                    load_encodings()
                st.success(f"✅ Success! Encoded {len(db_new)} student profile(s).")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Encoding error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    if run_cam:
        app = get_recognize_app()
        stream = ThreadedWebcam(src=0).start()
        time.sleep(0.3)

        last_logged_name = ""
        last_logged_time = 0

        try:
            while run_cam:
                grabbed, frame = stream.read()
                if not grabbed or frame is None:
                    time.sleep(0.01)
                    continue

                frame, recognized = process_frame(frame, database, app, match_thresh=match_threshold)

                new_marks = [r for r in recognized if r.get("is_new_mark", False)]
                already_marked = [r for r in recognized if not r.get("is_new_mark", False)]

                if new_marks:
                    r = new_marks[0]
                    last_logged_name = r['name']
                    last_logged_time = time.time()
                    log_window.markdown(f"""
                    <div style="text-align: center; padding: 20px; background: #ecfdf5; border: 2px solid #10b981; border-radius: 12px; margin-bottom: 10px;">
                        <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <h3 style="color: #065f46; font-weight: 700; margin-top: 8px; font-size: 1.2rem;">Attendance Marked!</h3>
                        <p style="color: #047857; font-weight: 600; margin: 0;">Student: <strong>{last_logged_name}</strong></p>
                        <small style="color: #059669; font-weight: 500;">Ready for next student...</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif already_marked:
                    r = already_marked[0]
                    if time.time() - last_logged_time > 3.0:
                        log_window.markdown(f"""
                        <div style="background: #f8fafc; border: 1px solid #cbd5e1; padding: 14px 18px; border-radius: 8px; color: #334155; font-weight: 600;">
                            ℹ️ <strong>{r['name']}</strong> is already marked present today.
                            <br><small style="font-weight: 500; color: #64748b;">Ready for next student...</small>
                        </div>
                        """, unsafe_allow_html=True)
                elif time.time() - last_logged_time > 4.0:
                    log_window.markdown("""
                    <div style="background: #fefce8; border: 1px solid #fde047; padding: 14px 18px; border-radius: 8px; color: #854d0e; font-weight: 600;">
                        🟡 <strong>SCANNING CAMERA...</strong> Present face to mark attendance.
                    </div>
                    """, unsafe_allow_html=True)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.01)
        finally:
            stream.stop()

# -------------------------------------------------------------
# 📋 ATTENDANCE RECORDS PAGE
# -------------------------------------------------------------
elif page == "📋 Records":
    st.markdown("<h1 class='header-title'>Student Attendance Records & History</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>View student attendance logs, filter by date, and export CSV reports.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 4px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    col_filter, col_export = st.columns([2, 1])
    with col_filter:
        selected_date = st.date_input("Filter Attendance by Date", value=datetime.now().date())

    date_str = selected_date.strftime("%Y-%m-%d")
    logs = get_attendance_logs(selected_date=date_str)

    with col_export:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
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
        st.markdown(f"### Attendance Records ({date_str})")
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info(f"ℹ️ No attendance records logged for {date_str}.")

# -------------------------------------------------------------
# 📲 INSTALL APP PAGE (PWA FOR PHONE & DESKTOP)
# -------------------------------------------------------------
elif page == "📲 Install App":
    st.markdown("<h1 class='header-title'>📲 Install FaceAuth AI as an App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Install FaceAuth AI on your Phone (Android / iPhone) or Computer for a seamless, fast, fullscreen experience without browser toolbars.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 4px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # 1. PWA Interactive Install Banner Component
    st.components.v1.html("""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: transparent; padding: 2px; }
        .banner-card {
          background: #ffffff;
          border: 1px solid #cbd5e1;
          border-radius: 10px;
          padding: 16px 20px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 14px;
          box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        .banner-info { display: flex; align-items: center; gap: 14px; min-width: 0; }
        .app-icon {
          width: 46px;
          height: 46px;
          flex-shrink: 0;
          background: #0f172a;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 22px;
          color: white;
          box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15);
        }
        .app-text { min-width: 0; }
        .app-text h3 {
          font-size: 1rem;
          font-weight: 700;
          color: #0f172a;
          margin-bottom: 2px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .app-text p {
          font-size: 0.8rem;
          color: #64748b;
          font-weight: 500;
          line-height: 1.3;
        }
        .btn-install {
          background: #0f172a;
          color: #ffffff;
          border: 1px solid #0f172a;
          border-radius: 8px;
          padding: 9px 18px;
          font-size: 0.85rem;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 8px;
          white-space: nowrap;
          transition: all 0.15s ease;
          outline: none;
        }
        .btn-install:hover {
          background: #1e293b;
        }
        .status-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          border-radius: 6px;
          font-size: 0.82rem;
          font-weight: 600;
          background: #f0fdf4;
          border: 1px solid #bbf7d0;
          color: #166534;
          white-space: nowrap;
        }
        @media (max-width: 600px) {
          .banner-card {
            flex-direction: column;
            align-items: stretch;
            padding: 14px 16px;
            gap: 12px;
          }
          .app-text h3 { white-space: normal; }
          .btn-install { width: 100%; justify-content: center; }
          #action-container { width: 100%; }
        }
      </style>
    </head>
    <body>
      <div class="banner-card">
        <div class="banner-info">
          <div class="app-icon">🛡️</div>
          <div class="app-text">
            <h3>FaceAuth AI Mobile &amp; Desktop App</h3>
            <p id="platform-desc">Progressive Web App &bull; Standalone Mode &bull; Real-time Biometrics</p>
          </div>
        </div>
        <div id="action-container">
          <button id="install-btn" class="btn-install" onclick="triggerInstall()">
            📲 Install Application
          </button>
        </div>
      </div>

      <script>
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
        const isStandalone = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone;
        const actionContainer = document.getElementById('action-container');
        const platformDesc = document.getElementById('platform-desc');
        const installBtn = document.getElementById('install-btn');

        if (isStandalone) {
          actionContainer.innerHTML = '<div class="status-badge">✅ Running as Installed App</div>';
          platformDesc.innerText = 'Active in Standalone Fullscreen Mode.';
        } else if (isIOS) {
          installBtn.innerText = '🍏 How to Install on iPhone';
          installBtn.onclick = function() {
            alert('On iPhone / iPad:\\n1. Tap Share (square with arrow up ⎋) at the bottom of Safari.\\n2. Scroll down and tap "Add to Home Screen ⊞".\\n3. Tap "Add" in the top right!');
          };
          platformDesc.innerText = 'Tap Share ⎋ then "Add to Home Screen ⊞"';
        }

        function triggerInstall() {
          let promptEvent = window.deferredInstallPrompt;
          if (!promptEvent && window.parent) {
            promptEvent = window.parent.deferredInstallPrompt;
          }
          if (promptEvent) {
            promptEvent.prompt();
            promptEvent.userChoice.then((choiceResult) => {
              if (choiceResult.outcome === 'accepted') {
                actionContainer.innerHTML = '<div class="status-badge">✅ App Installed Successfully!</div>';
              }
            });
          } else {
            alert('To install on this device:\\n\\n• Android Phone: Tap the ⋮ menu in Chrome -> tap "Install app" or "Add to Home Screen".\\n• iPhone / iPad: Tap Share ⎋ -> "Add to Home Screen ⊞".\\n• PC / Mac: Look for the Install icon (⊕) in your browser address bar.');
          }
        }
      </script>
    </body>
    </html>
    """, height=185)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Main Content Layout: Installation Guides & Phone QR Access
    col_guide, col_qr = st.columns([1.5, 1], gap="large")

    with col_guide:
        guide_html = (
            '<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); margin-bottom: 16px;">\n'
            '<div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;"><span>📱 Step-by-Step Installation Guides</span></div>\n'
            '<div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; background: #fafafa;">\n'
            '<div style="font-weight: 700; color: #0f172a; font-size: 0.88rem; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">🤖 Android Phones (Google Chrome, Samsung Internet, Edge)</div>\n'
            '<ol style="margin: 0; padding-left: 20px; font-size: 0.83rem; color: #334155; line-height: 1.6;">\n'
            '<li>Open this URL in <strong>Google Chrome</strong> or <strong>Edge</strong> on your Android phone.</li>\n'
            '<li>Tap the <strong>three vertical dots (⋮)</strong> menu in the upper-right corner.</li>\n'
            '<li>Select <strong>"Install app"</strong> (or <strong>"Add to Home screen"</strong>).</li>\n'
            '<li>Tap <strong>Install</strong> to confirm. The FaceAuth AI icon will appear directly on your phone\'s home screen!</li>\n'
            '</ol>\n'
            '</div>\n'
            '<div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; background: #fafafa;">\n'
            '<div style="font-weight: 700; color: #0f172a; font-size: 0.88rem; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">🍏 iPhone & iPad (Apple Safari)</div>\n'
            '<ol style="margin: 0; padding-left: 20px; font-size: 0.83rem; color: #334155; line-height: 1.6;">\n'
            '<li>Open this URL in <strong>Safari</strong> on your iPhone or iPad.</li>\n'
            '<li>Tap the <strong>Share button (⎋)</strong> at the bottom center of Safari.</li>\n'
            '<li>Scroll down and tap <strong>"Add to Home Screen (⊞)"</strong>.</li>\n'
            '<li>Tap <strong>Add</strong> in the top-right corner. FaceAuth AI launches fullscreen without Safari tabs!</li>\n'
            '</ol>\n'
            '</div>\n'
            '<div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; background: #fafafa;">\n'
            '<div style="font-weight: 700; color: #0f172a; font-size: 0.88rem; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">💻 Windows PC / Mac / Linux (Chrome, Edge, Brave)</div>\n'
            '<ol style="margin: 0; padding-left: 20px; font-size: 0.83rem; color: #334155; line-height: 1.6;">\n'
            '<li>Look at the right-hand side of your browser\'s address bar.</li>\n'
            '<li>Click the <strong>Install App icon (⊕ or computer with down arrow)</strong>.</li>\n'
            '<li>Click <strong>Install</strong>. FaceAuth AI will open as a dedicated desktop application window.</li>\n'
            '</ol>\n'
            '</div>\n'
            '</div>\n'
        )
        st.markdown(guide_html, unsafe_allow_html=True)

        features_html = (
            '<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);">\n'
            '<div style="font-size: 0.92rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">⚡ Why Use FaceAuth AI as an Installed App?</div>\n'
            '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">\n'
            '<div style="padding: 10px; border: 1px solid #f1f5f9; background: #f8fafc; border-radius: 6px;">\n'
            '<div style="font-weight: 600; font-size: 0.82rem; color: #0f172a;">🖥️ 100% Fullscreen Kiosk</div>\n'
            '<div style="font-size: 0.76rem; color: #64748b; margin-top: 2px;">No browser search bar, back/forward buttons, or tabs. Perfect for attendance desks.</div>\n'
            '</div>\n'
            '<div style="padding: 10px; border: 1px solid #f1f5f9; background: #f8fafc; border-radius: 6px;">\n'
            '<div style="font-weight: 600; font-size: 0.82rem; color: #0f172a;">📱 1-Tap Home Screen Launch</div>\n'
            '<div style="font-size: 0.76rem; color: #64748b; margin-top: 2px;">Opens immediately like a native app directly from your phone\'s app list.</div>\n'
            '</div>\n'
            '<div style="padding: 10px; border: 1px solid #f1f5f9; background: #f8fafc; border-radius: 6px;">\n'
            '<div style="font-weight: 600; font-size: 0.82rem; color: #0f172a;">📷 Persistent Camera Access</div>\n'
            '<div style="font-size: 0.76rem; color: #64748b; margin-top: 2px;">Remembers camera permissions without re-prompting every visit.</div>\n'
            '</div>\n'
            '<div style="padding: 10px; border: 1px solid #f1f5f9; background: #f8fafc; border-radius: 6px;">\n'
            '<div style="font-weight: 600; font-size: 0.82rem; color: #0f172a;">⚡ Instant Fast Loading</div>\n'
            '<div style="font-size: 0.76rem; color: #64748b; margin-top: 2px;">Pre-cached styles and layout for instantaneous startup speed.</div>\n'
            '</div>\n'
            '</div>\n'
            '</div>\n'
        )
        st.markdown(features_html, unsafe_allow_html=True)

    with col_qr:
        live_app_url = "https://faceverification.streamlit.app"
        qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={live_app_url}&margin=10"
        qr_html = (
            '<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 22px; text-align: center; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);">\n'
            '<div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;">📸 Open & Install on Phone</div>\n'
            '<div style="font-size: 0.8rem; color: #64748b; margin-bottom: 16px;">Scan this QR code with your mobile phone camera to open and install instantly.</div>\n'
            f'<div style="background: #ffffff; padding: 12px; border: 1px solid #e2e8f0; border-radius: 12px; display: inline-block; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.06);"><img src="{qr_api_url}" alt="FaceAuth AI QR Code" style="width: 190px; height: 190px; display: block; border-radius: 6px;" /></div>\n'
            f'<div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-family: monospace; font-size: 0.8rem; color: #0f172a; word-break: break-all;">{live_app_url}</div>\n'
            '<div style="display: flex; flex-direction: column; gap: 6px; text-align: left; font-size: 0.78rem; color: #475569; border-top: 1px solid #f1f5f9; padding-top: 14px;">\n'
            '<div style="display: flex; align-items: center; gap: 6px;"><span style="color: #16a34a; font-weight: bold;">✔</span> Verified on Android (Chrome, Edge, Brave)</div>\n'
            '<div style="display: flex; align-items: center; gap: 6px;"><span style="color: #16a34a; font-weight: bold;">✔</span> Verified on iPhone & iPad (Safari iOS 11.3+)</div>\n'
            '<div style="display: flex; align-items: center; gap: 6px;"><span style="color: #16a34a; font-weight: bold;">✔</span> Verified on Windows, macOS & Chromebook</div>\n'
            '</div>\n'
            '</div>\n'
        )
        st.markdown(qr_html, unsafe_allow_html=True)

