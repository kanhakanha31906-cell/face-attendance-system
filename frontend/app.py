import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# 1. Page Configuration (Mobile-Optimized)
st.set_page_config(
    page_title="Face Attendance",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Safe Backend API URL Detection
API_URL = "http://127.0.0.1:8000"
try:
    if "API_URL" in st.secrets:
        API_URL = st.secrets["API_URL"]
except Exception:
    pass
API_URL = os.getenv("API_URL", API_URL).rstrip("/")

# 3. Mobile-First Responsive CSS
st.markdown("""
    <style>
    /* Clean container padding for phones */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    
    /* Responsive rounded camera view */
    [data-testid="stCameraInput"] video {
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        width: 100% !important;
    }

    /* Thumb-friendly full-width buttons */
    div.stButton > button:first-child {
        width: 100%;
        height: 3rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 10px;
        background-color: #2563eb;
        color: white;
        border: none;
        margin-top: 0.5rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* Tab sizing */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    /* Scrollable dataframes on small screens */
    [data-testid="stDataFrame"] {
        width: 100% !important;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

# 4. App Header
st.title("📸 Smart Attendance")
st.caption(f"Connected to: `{API_URL}`")

# 5. Mobile Tab Navigation
tab_scan, tab_register, tab_records = st.tabs(["📷 Check-In", "👤 Enroll", "📊 Records"])

# --- TAB 1: MARK ATTENDANCE ---
with tab_scan:
    st.subheader("Mark Today's Attendance")
    camera_img = st.camera_input("Take a photo to check in", key="attendance_cam")

    if camera_img is not None:
        if st.button("Verify & Submit Attendance", key="btn_attendance"):
            with st.spinner("Analyzing face..."):
                try:
                    files = {"file": (camera_img.name, camera_img.getvalue(), "image/jpeg")}
                    res = requests.post(f"{API_URL}/mark-attendance/", files=files, timeout=60)
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"✅ Verified: **{data.get('name', 'User')}** ({data.get('user_id', '')})")
                        st.info(f"Logged at: {data.get('timestamp', datetime.now().strftime('%H:%M:%S'))}")
                    elif res.status_code == 400:
                        st.warning(f"⚠️ {res.json().get('detail', 'Attendance already marked or face issue.')}")
                    elif res.status_code == 404:
                        st.error("❌ Face not recognized. Please enroll first.")
                    else:
                        st.error(f"Error ({res.status_code}): {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend server unreachable. If on free tier, please wait 40 seconds for it to wake up.")
                except Exception as e:
                    st.error(f"Failed to submit: {str(e)}")

# --- TAB 2: REGISTER NEW USER ---
with tab_register:
    st.subheader("Enroll New Face")
    new_name = st.text_input("Full Name", placeholder="e.g. John Doe")
    new_id = st.text_input("User ID / Roll No", placeholder="e.g. EMP001")
    reg_img = st.camera_input("Capture profile photo", key="reg_cam")

    if st.button("Complete Registration", key="btn_reg"):
        if not new_name.strip() or not new_id.strip():
            st.error("Please provide both Name and User ID.")
        elif reg_img is None:
            st.error("Please capture a photo first.")
        else:
            with st.spinner("Generating face embeddings..."):
                try:
                    files = {"file": (reg_img.name, reg_img.getvalue(), "image/jpeg")}
                    data = {"user_id": new_id.strip(), "name": new_name.strip()}
                    res = requests.post(f"{API_URL}/register/", files=files, data=data, timeout=60)

                    if res.status_code == 200:
                        st.success(f"🎉 Successfully enrolled **{new_name}**!")
                    else:
                        st.error(f"Registration failed: {res.json().get('detail', res.text)}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend server unreachable. Please wait 40 seconds for it to wake up.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# --- TAB 3: ATTENDANCE RECORDS ---
with tab_records:
    st.subheader("Attendance History")
    
    if st.button("🔄 Refresh Records", key="btn_refresh"):
        st.rerun()

    try:
        res = requests.get(f"{API_URL}/attendance/", timeout=30)
        if res.status_code == 200:
            records = res.json()
            if records:
                df = pd.DataFrame(records)
                # Display clean view
                st.dataframe(df, use_container_width=True)
                
                # CSV Download Button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv,
                    file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No attendance records logged yet.")
        else:
            st.warning("Could not fetch attendance records.")
    except Exception:
        st.warning("Connect to your backend to view records.")