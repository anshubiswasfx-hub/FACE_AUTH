import os
from database.database import get_db_connection, create_database
from utils.face_detector import capture_faces

def register_student(name, enrollment, branch, capture_callback=None):
    """
    Registers a new student, captures face dataset, and writes to database.
    Returns (success: bool, message: str)
    """
    create_database()
    name = name.strip()
    enrollment = enrollment.strip()
    branch = branch.strip()

    if not name or not enrollment or not branch:
        return False, "All fields (Name, Enrollment, Branch) are required!"

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE roll_no = ?", (enrollment,))
    if cursor.fetchone():
        conn.close()
        return False, f"Enrollment Number '{enrollment}' already exists in database!"

    student_folder = os.path.join("dataset", f"{enrollment}_{name.replace(' ', '_')}")
    os.makedirs(student_folder, exist_ok=True)

    if capture_callback:
        success = capture_callback(student_folder)
    else:
        success = capture_faces(student_folder)

    if not success:
        conn.close()
        return False, "Face capture was cancelled or failed to capture required images."

    try:
        cursor.execute(
            """
            INSERT INTO students (name, roll_no, department, image_folder)
            VALUES (?, ?, ?, ?)
            """,
            (name, enrollment, branch, student_folder)
        )
        conn.commit()
        conn.close()
        return True, f"✅ Student '{name}' ({enrollment}) registered successfully!"
    except Exception as e:
        conn.close()
        return False, f"Database insertion failed: {str(e)}"

def run_cli_register():
    print("\n========== Student Registration ==========\n")
    name = input("Student Name        : ")
    enrollment = input("Enrollment Number   : ")
    branch = input("Branch              : ")

    success, msg = register_student(name, enrollment, branch)
    print(f"\n{msg}")

if __name__ == "__main__":
    run_cli_register()