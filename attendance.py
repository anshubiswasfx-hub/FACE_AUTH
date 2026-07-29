from datetime import datetime
import os
import pandas as pd
from database.database import get_db_connection, get_student_by_roll_no, create_database
from config import REPORTS_DIR

def mark_attendance(roll_no):
    """
    Marks attendance for a student with given roll_no for the current date.
    Returns (success: bool, status_message: str, student_details: dict)
    """
    create_database()
    student = get_student_by_roll_no(roll_no)
    if not student:
        return False, f"Student with roll number '{roll_no}' not found in database.", None

    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND date = ?",
            (student["id"], today_date)
        )
        existing = cursor.fetchone()

        if existing:
            conn.close()
            return False, f"Attendance already marked for {student['name']} today at {existing['time']}.", student

        cursor.execute(
            """
            INSERT INTO attendance (student_id, date, time, status)
            VALUES (?, ?, ?, 'Present')
            """,
            (student["id"], today_date, current_time)
        )
        conn.commit()
        conn.close()
        return True, f"✅ Attendance marked for {student['name']} ({student['roll_no']}) at {current_time}.", student
    except Exception as e:
        conn.close()
        return False, f"Error marking attendance: {str(e)}", student

def get_attendance_logs(selected_date=None):
    """
    Fetches attendance logs joined with student details.
    """
    create_database()
    conn = get_db_connection()
    cursor = conn.cursor()

    if selected_date:
        query = """
        SELECT a.id, s.roll_no, s.name, s.department, a.date, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.time DESC
        """
        cursor.execute(query, (selected_date,))
    else:
        query = """
        SELECT a.id, s.roll_no, s.name, s.department, a.date, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        ORDER BY a.date DESC, a.time DESC
        """
        cursor.execute(query)

    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def export_attendance_report(selected_date=None):
    """
    Exports attendance logs to a CSV file in REPORTS_DIR and returns DataFrame.
    """
    records = get_attendance_logs(selected_date)
    df = pd.DataFrame(records)

    if not df.empty:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        filename = f"attendance_{selected_date or 'all'}.csv"
        file_path = os.path.join(REPORTS_DIR, filename)
        df.to_csv(file_path, index=False)
        print(f"📊 Report saved to {file_path}")

    return df

if __name__ == "__main__":
    logs = get_attendance_logs()
    print(f"Total attendance records: {len(logs)}")
