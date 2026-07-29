import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "students.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        image_folder TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        status TEXT DEFAULT 'Present',
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
        UNIQUE(student_id, date)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS spoof_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        reason TEXT NOT NULL,
        liveness_score REAL NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    try:
        print("[OK] Database created/verified successfully!")
    except UnicodeEncodeError:
        print("[OK] Database created/verified successfully!")

def reset_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS attendance")
    cursor.execute("DROP TABLE IF EXISTS students")
    cursor.execute("DROP TABLE IF EXISTS spoof_logs")
    conn.commit()
    conn.close()
    create_database()
    try:
        print("[OK] Database reset successfully!")
    except UnicodeEncodeError:
        print("[OK] Database reset successfully!")

def get_all_students():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY id DESC")
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return students

def get_student_by_roll_no(roll_no):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE roll_no = ?", (roll_no,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def log_spoof_attempt(reason, liveness_score):
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO spoof_logs (date, time, reason, liveness_score)
            VALUES (?, ?, ?, ?)
            """,
            (today_date, current_time, reason, float(liveness_score))
        )
        conn.commit()
    except Exception as e:
        print(f"Error logging spoof attempt: {e}")
    finally:
        conn.close()

def get_spoof_logs(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM spoof_logs ORDER BY id DESC LIMIT ?", (limit,))
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs

if __name__ == "__main__":
    create_database()