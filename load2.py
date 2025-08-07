import cv2
import face_recognition
import numpy as np
import datetime
import time
import openpyxl
import os
import pickle

def track_attendance(known_face_encodings, known_face_names, model_path="face_recognition_model.pkl"):
    """
    Tracks attendance using facial recognition from a video feed.

    Args:
        known_face_encodings (list): List of known face encodings.
        known_face_names (list): List of known face names.
        model_path (str, optional): Path to the saved model file. Defaults to "face_recognition_model.pkl".
    """

    # Video capture
    video_capture = cv2.VideoCapture(0)

    # Attendance Tracking
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Attendance Log"
    sheet.append(["Name", "Time", "Date", "Attendance Percentage", "Status"])

    # Final Output Sheet
    final_sheet = workbook.create_sheet(title="Final Attendance")
    final_sheet.append(["Name", "Date", "Time", "Attendance Percentage", "Attendance Status", "Status"])

    attendance_data = {}

    # Time settings
    total_time = 60  
    slice_duration = 15 
    start_time = time.time()
    slice_start_times = [start_time + i * slice_duration for i in range(4)]
    slice_attendance = {i: set() for i in range(4)}

    print("Starting facial recognition...")
    while time.time() - start_time < total_time:
        current_time = time.time()
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            name = "Unknown"
            if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                name = known_face_names[best_match_index]

                now = datetime.datetime.now()
                current_time_str = now.strftime("%H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")

                # Determine the slice index
                for i, slice_start in enumerate(slice_start_times):
                    if slice_start <= current_time < slice_start + slice_duration:
                        if name not in slice_attendance[i]:
                            slice_attendance[i].add(name)

                            # Calculate attendance percentage
                            total_slices_present = sum(1 for s in slice_attendance.values() if name in s)
                            attendance_percentage = (total_slices_present / 4) * 100

                            # Determine late status
                            status = "On Time" if name in slice_attendance[0] else "Late"

                            sheet.append([name, current_time_str, date_str, f"{attendance_percentage:.2f}%", status])
                            print(f"{name} attended in slice {i + 1} at {current_time_str}, {date_str}")
                        break

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save final attendance summary
    for name in set(known_face_names):
        total_slices_present = sum(1 for s in slice_attendance.values() if name in s)
        attendance_percentage = (total_slices_present / 4) * 100

        # Determine attendance status
        if total_slices_present >= 2:
            attendance_status = "Present"
            status = "On Time" if name in slice_attendance[0] else "Late"
        elif total_slices_present == 1:
            attendance_status = "Present"
            status = "Late"
        else:
            attendance_percentage = 0
            attendance_status = "Absent"
            status = "Absent"

        final_sheet.append([name, date_str, current_time_str, f"{attendance_percentage:.2f}%", attendance_status, status])

    # Save attendance
    if not os.path.exists("db"):
        os.makedirs("db")

    workbook.save("db/attendance_60_seconds.xlsx")
    print("Attendance data saved.")

    video_capture.release()
    cv2.destroyAllWindows()

# Load the model
try:
    with open("face_recognition_model.pkl", "rb") as f:
        loaded_model_data = pickle.load(f)
        known_face_encodings = loaded_model_data["encodings"]
        known_face_names = loaded_model_data["names"]

    # Run the attendance tracking
    track_attendance(known_face_encodings, known_face_names)

except FileNotFoundError:
  print("Error: face_recognition_model.pkl not found. Make sure to train your model first.")