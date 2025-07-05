# Student Movement Detection - Project Documentation

## Overview

This project is a classroom monitoring system that uses face recognition and computer vision to detect student distraction in real-time. It leverages state-of-the-art AI models and provides a web dashboard for visualization.

---

## Tech Stack

- **Programming Language:** Python
- **Web Framework:** Flask
- **Frontend:** HTML5, Bootstrap 5, JavaScript (with Plotly for data visualization)
- **Computer Vision:** OpenCV (cv2)
- **Object Detection:** YOLO (via ultralytics package, e.g., YOLOv8)
- **Face and Pose Estimation:** MediaPipe
- **Data Handling:** JSON (for storing distraction data)
- **Other Libraries:** NumPy, os, time

---

## Features

- **Real-time distraction detection:** Analyzes webcam video, detects students, and identifies distraction (talking or head turning).
- **Dashboard:** Web interface showing:
    - Total students detected
    - Number of distracted students
    - Average focus level
    - Table of individual distraction records, including images
- **Image capture:** Captures and stores images of distracted students.
- **Persistent storage:** Saves distraction data and images in the `static/` directory.

---

## Main Components

### 1. acc.py

- **Function:** Runs real-time detection using your webcam.
- **Workflow:**
    - Loads YOLO model for student detection.
    - Uses MediaPipe for face mesh and pose detection.
    - For each detected student:
        - Detects talking (lip movement) and head turning (pose estimation).
        - Captures image and updates distraction count if distracted.
    - Writes status to `static/student_distraction_data.json`.

### 2. main.py

- **Function:** Flask-based web server.
- **Routes:**
    - `/` – Renders the dashboard (dashboard.html).
    - `/data` – Returns distraction data as JSON.

### 3. templates/dashboard.html

- **Function:** Frontend dashboard user interface.
- **Features:**
    - Displays statistics and student records.
    - Uses JavaScript to poll `/data` endpoint every 2 seconds for live updates.

---

## Data Structure

- **JSON File:** `static/student_distraction_data.json`
    ```json
    {
      "student_id": {
        "talking": int,
        "turning": int,
        "image": "path/to/image.jpg"
      },
      ...
    }
    ```

---

## How to Run

1. **Install dependencies:**  
   - Python, OpenCV, Flask, ultralytics, mediapipe, numpy, etc.
   - Example:  
     ```
     pip install flask opencv-python ultralytics mediapipe numpy
     ```
2. **Start detection:**  
   - Run `acc.py` to analyze video and collect data.
3. **Start dashboard:**  
   - Run `main.py` to start Flask web server.
   - Open browser at `http://localhost:5000` to view the dashboard.

---

## Example Workflow

1. The system captures webcam feed and analyzes each student for distraction.
2. Distraction events update counters and save images.
3. The dashboard displays live stats and records.

---

## Project Structure

```
student_face_recognition-/
├── acc.py                # Real-time detection script
├── main.py               # Flask web server
├── static/
│   ├── student_distraction_data.json
│   └── distracted_students/    # Saved images
└── templates/
    └── dashboard.html    # Dashboard UI
```

---

## Notes

- This project is for demonstration/educational use.
- It detects distraction using video analysis, but does not identify students by name.
- Requires a compatible webcam and the necessary Python dependencies.
