import streamlit as st
import requests
import pandas as pd

if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]
else:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("Smart Face Recognition Attendance System")

tabs = st.tabs(["Check-In Camera", "User Registration", "Attendance Records"])

# Tab 1: Check-in
with tabs[0]:
    st.subheader("Face Check-In")
    img_capture = st.camera_input("Look at the camera to check in")
    
    if img_capture is not None:
        files = {"file": ("checkin.jpg", img_capture.getvalue(), "image/jpeg")}
        with st.spinner("Analyzing face..."):
            res = requests.post(f"{API_URL}/verify", files=files)
            if res.status_code == 200:
                data = res.json()
                if data.get("matched"):
                    st.success(f"{data['message']} (Confidence: {data['score']})")
                else:
                    st.error(data.get("message"))
            else:
                st.warning(res.json().get("detail", "Error verifying image."))

# Tab 2: Enrollment
with tabs[1]:
    st.subheader("Enroll New Employee / Student")
    with st.form("enrollment_form", clear_on_submit=True):
        user_id = st.text_input("User ID (e.g., EMP001, ROLL42)")
        name = st.text_input("Full Name")
        upload_img = st.file_uploader("Upload Profile Image", type=["jpg", "jpeg", "png"])
        submit_btn = st.form_submit_button("Register Face")

        if submit_btn:
            if not user_id or not name or not upload_img:
                st.error("All fields are required.")
            else:
                files = {"file": (upload_img.name, upload_img.getvalue(), upload_img.type)}
                data = {"user_id": user_id, "name": name}
                res = requests.post(f"{API_URL}/enroll", data=data, files=files)
                if res.status_code == 200:
                    st.success(res.json().get("message"))
                else:
                    st.error(res.json().get("detail", "Failed to enroll user."))

# Tab 3: Reports
# Tab 3: Records Table & Export
# Tab 3: Records Table, Date Filter & Export
# Tab 3: Records Table, Multi-Filter (Date, Name, User ID) & CSV Export
with tabs[2]:
    st.subheader("Attendance History & Reports")

    try:
        res = requests.get(f"{API_URL}/attendance-logs")
        if res.status_code == 200:
            records = res.json()
            if records:
                df = pd.DataFrame(records)

                # Ensure timestamps and dates are formatted properly
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["record_date"] = df["timestamp"].dt.date

                # Base boundaries for date filter
                min_date = df["record_date"].min()
                max_date = df["record_date"].max()

                # Filter Controls Row
                col_search, col_date = st.columns([2, 2])

                with col_search:
                    search_query = st.text_input(
                        "🔍 Search by Name or User ID:",
                        placeholder="Type to filter (e.g. John, EMP01)..."
                    ).strip()

                with col_date:
                    date_range = st.date_input(
                        "📅 Filter by Date Range:",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        help="Select start and end dates"
                    )

                # 1. Apply Date Filter
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = df[
                        (df["record_date"] >= start_date) & 
                        (df["record_date"] <= end_date)
                    ]
                elif isinstance(date_range, tuple) and len(date_range) == 1:
                    start_date = end_date = date_range[0]
                    filtered_df = df[df["record_date"] == start_date]
                else:
                    start_date, end_date = min_date, max_date
                    filtered_df = df.copy()

                # 2. Apply Text/ID Filter (Case-Insensitive)
                if search_query:
                    search_mask = (
                        filtered_df["name"].str.contains(search_query, case=False, na=False) |
                        filtered_df["user_id"].str.contains(search_query, case=False, na=False)
                    )
                    filtered_df = filtered_df[search_mask]

                # Prepare Clean Export/Display Table
                display_cols = ["id", "name", "user_id", "timestamp"]
                available_cols = [c for c in display_cols if c in filtered_df.columns]
                export_df = filtered_df[available_cols].copy()
                export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

                # Results Summary
                st.caption(
                    f"Showing **{len(export_df)}** entries found "
                    f"(Date range: {start_date} to {end_date}" +
                    (f", Search: \"{search_query}\"" if search_query else "") + ")"
                )

                # Download Button for Filtered Results
                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download Filtered CSV ({len(export_df)} rows)",
                    data=csv_bytes,
                    file_name=f"attendance_report_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )

                # Render Table
                st.dataframe(export_df, use_container_width=True)
            else:
                st.info("No attendance records found yet.")
        else:
            st.error("Failed to retrieve attendance logs from backend.")
    except requests.exceptions.ConnectionError:
        st.error("Backend server is offline! Start FastAPI first.")