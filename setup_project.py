import os

# Define folders and their respective files with contents
files = {
    # 1. Database init
    "backend/database/__init__.py": "",
    
    # 2. Database connection & schema
    "backend/database/connection.py": '''import sqlite3

DB_PATH = "attendance.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        date TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        UNIQUE(user_id, date)
    )
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
''',

    # 3. Services init
    "backend/services/__init__.py": "",

    # 4. Face recognition engine
    "backend/services/face_engine.py": '''import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, model_name: str = "buffalo_sc"):
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_embedding(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, "Invalid image format."

        faces = self.app.get(frame)
        if len(faces) == 0:
            return None, "No face detected in the image."
        if len(faces) > 1:
            return None, "Multiple faces detected. Ensure only one face is visible."
        
        embedding = faces[0].normed_embedding
        return embedding.astype(np.float32), None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

face_engine = FaceEngine()
''',

    # 5. FastAPI main
    "backend/main.py": '''from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from datetime import datetime
import numpy as np
from backend.database.connection import get_db, init_db
from backend.services.face_engine import face_engine

app = FastAPI(title="Face Attendance System API")

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/enroll")
async def enroll_user(
    user_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    embedding, error = face_engine.extract_embedding(image_bytes)
    if error:
        raise HTTPException(status_code=400, detail=error)

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (user_id, name, embedding) VALUES (?, ?, ?)",
            (user_id, name, embedding.tobytes())
        )
        conn.commit()
    except Exception:
        raise HTTPException(status_code=400, detail="User ID already exists.")
    finally:
        conn.close()

    return {"status": "success", "message": f"User '{name}' enrolled successfully."}

@app.post("/verify")
async def verify_and_log(file: UploadFile = File(...)):
    image_bytes = await file.read()
    query_emb, error = face_engine.extract_embedding(image_bytes)
    if error:
        raise HTTPException(status_code=400, detail=error)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name, embedding FROM users")
    registered_users = cursor.fetchall()

    best_match = None
    highest_score = -1.0
    SIMILARITY_THRESHOLD = 0.55

    for user in registered_users:
        stored_emb = np.frombuffer(user["embedding"], dtype=np.float32)
        score = face_engine.cosine_similarity(query_emb, stored_emb)
        if score > highest_score:
            highest_score = score
            best_match = user

    if not best_match or highest_score < SIMILARITY_THRESHOLD:
        conn.close()
        return {"matched": False, "message": "Unknown person. Face not recognized."}

    today_date = datetime.now().strftime("%Y-%m-%d")
    try:
        cursor.execute(
            "INSERT INTO attendance (user_id, date) VALUES (?, ?)",
            (best_match["user_id"], today_date)
        )
        conn.commit()
        status_msg = f"Attendance marked for {best_match['name']}!"
    except Exception:
        status_msg = f"Attendance was already marked today for {best_match['name']}."
    finally:
        conn.close()

    return {
        "matched": True,
        "user_id": best_match["user_id"],
        "name": best_match["name"],
        "score": round(highest_score, 3),
        "message": status_msg
    }

@app.get("/attendance-logs")
def get_logs():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id, u.name, a.user_id, a.timestamp, a.date 
        FROM attendance a
        JOIN users u ON a.user_id = u.user_id
        ORDER BY a.timestamp DESC
    """)
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs
''',

    # 6. Streamlit Frontend
    "frontend/app.py": '''import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("Smart Face Recognition Attendance System")

tabs = st.tabs(["Check-In Camera", "Register New User", "Attendance History"])

with tabs[0]:
    st.subheader("Look into the camera to mark attendance")
    img_capture = st.camera_input("Face Scan")
    
    if img_capture is not None:
        files = {"file": ("checkin.jpg", img_capture.getvalue(), "image/jpeg")}
        with st.spinner("Analyzing face..."):
            try:
                res = requests.post(f"{API_URL}/verify", files=files)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("matched"):
                        st.success(f"{data['message']} (Confidence: {data['score']})")
                    else:
                        st.error(data.get("message"))
                else:
                    st.warning(res.json().get("detail", "Error processing face."))
            except requests.exceptions.ConnectionError:
                st.error("Backend server is offline! Start FastAPI first.")

with tabs[1]:
    st.subheader("Enroll New Person")
    with st.form("enroll_form", clear_on_submit=True):
        user_id = st.text_input("User ID (e.g., EMP01, STU101)")
        name = st.text_input("Full Name")
        upload_img = st.file_uploader("Upload clear frontal photo", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Register")

        if submit:
            if not user_id or not name or not upload_img:
                st.error("Please fill all fields and provide an image.")
            else:
                files = {"file": (upload_img.name, upload_img.getvalue(), upload_img.type)}
                data = {"user_id": user_id, "name": name}
                try:
                    res = requests.post(f"{API_URL}/enroll", data=data, files=files)
                    if res.status_code == 200:
                        st.success(res.json().get("message"))
                    else:
                        st.error(res.json().get("detail", "Failed to register."))
                except requests.exceptions.ConnectionError:
                    st.error("Backend server is offline!")

with tabs[2]:
    st.subheader("Today's Attendance Logs")
    if st.button("Refresh Table"):
        try:
            res = requests.get(f"{API_URL}/attendance-logs")
            if res.status_code == 200:
                records = res.json()
                if records:
                    df = pd.DataFrame(records)
                    st.dataframe(df[["id", "name", "user_id", "timestamp"]], use_container_width=True)
                else:
                    st.info("No attendance records found yet.")
        except requests.exceptions.ConnectionError:
            st.error("Backend server is offline!")
'''
}

# Generate directories and write files
for path, content in files.items():
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path}")

print("\nProject structure generated successfully!")