import sqlite3

DB_PATH = "attendance.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Registered users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Attendance log table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        date TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        UNIQUE(user_id, date)  -- Enforces 1 attendance record per user per day
    )
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()