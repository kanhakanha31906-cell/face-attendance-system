from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

    return {"status": "success", "message": f"User {name} enrolled successfully."}

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
    SIMILARITY_THRESHOLD = 0.60  # Typical operating threshold for ArcFace / buffalo_sc

    for user in registered_users:
        stored_emb = np.frombuffer(user["embedding"], dtype=np.float32)
        score = face_engine.cosine_similarity(query_emb, stored_emb)
        if score > highest_score:
            highest_score = score
            best_match = user

    if not best_match or highest_score < SIMILARITY_THRESHOLD:
        conn.close()
        return {"matched": False, "message": "Unknown face detected."}

    # Log attendance
    today_date = datetime.now().strftime("%Y-%m-%d")
    try:
        cursor.execute(
            "INSERT INTO attendance (user_id, date) VALUES (?, ?)",
            (best_match["user_id"], today_date)
        )
        conn.commit()
        status_msg = f"Attendance recorded for {best_match['name']}."
    except Exception:
        status_msg = f"Attendance already recorded today for {best_match['name']}."
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