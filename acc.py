import cv2
import time
import os
import json
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Constants
FRAME_SKIP = 2  # Process every 2nd frame for performance
CONFIDENCE_THRESHOLD = 0.6  # Higher confidence threshold for YOLO
OUTPUT_FILE = "classroom_recording.mp4"

# Open webcam
cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = 30

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))

# Load YOLO Model (use a larger model for better accuracy)
model = YOLO("yolov8m.pt")

# MediaPipe Components
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Track student distractions
DATA_FILE = "static/student_distraction_data.json"
if not os.path.exists("static"):
    os.makedirs("static")

student_distraction_count = {}

# Create folder for distracted images
DISTRACTED_DIR = "static/distracted_students"
if not os.path.exists(DISTRACTED_DIR):
    os.makedirs(DISTRACTED_DIR)

frame_count = 0

# Function to calculate head pose
def calculate_head_pose(face_landmarks, frame_shape):
    # Extract key landmarks for head pose estimation
    image_points = np.array([
        (face_landmarks.landmark[1].x * frame_shape[1], face_landmarks.landmark[1].y * frame_shape[0]),  # Nose tip
        (face_landmarks.landmark[33].x * frame_shape[1], face_landmarks.landmark[33].y * frame_shape[0]),  # Chin
        (face_landmarks.landmark[61].x * frame_shape[1], face_landmarks.landmark[61].y * frame_shape[0]),  # Left eye corner
        (face_landmarks.landmark[291].x * frame_shape[1], face_landmarks.landmark[291].y * frame_shape[0]),  # Right eye corner
        (face_landmarks.landmark[199].x * frame_shape[1], face_landmarks.landmark[199].y * frame_shape[0]),  # Left mouth corner
        (face_landmarks.landmark[425].x * frame_shape[1], face_landmarks.landmark[425].y * frame_shape[0]),  # Right mouth corner
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ])

    # Camera matrix (assume no lens distortion)
    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    # SolvePnP to estimate head pose
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)

    # Convert rotation vector to angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if conf > CONFIDENCE_THRESHOLD and cls == 0:  # Detect people
                if track_id not in student_distraction_count:
                    student_distraction_count[track_id] = {"talking": 0, "turning": 0, "image": ""}

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Student {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                face_region = frame[y1:y2, x1:x2]
                if face_region.size == 0:
                    continue

                rgb_frame = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                # Lip Movement Detection
                face_results = face_mesh.process(rgb_frame)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        h, w, _ = face_region.shape
                        upper_lip = face_landmarks.landmark[13]
                        lower_lip = face_landmarks.landmark[14]
                        lip_distance = abs(lower_lip.y - upper_lip.y) * h

                        if lip_distance > 5:  # Talking threshold
                            student_distraction_count[track_id]["talking"] += 1
                            img_path = f"{DISTRACTED_DIR}/student_{track_id}.jpg"
                            cv2.imwrite(img_path, face_region)
                            student_distraction_count[track_id]["image"] = img_path

                        # Head Pose Estimation
                        angles = calculate_head_pose(face_landmarks, (h, w))
                        if abs(angles[1]) > 15:  # Head turning threshold
                            student_distraction_count[track_id]["turning"] += 1
                            img_path = f"{DISTRACTED_DIR}/student_{track_id}.jpg"
                            cv2.imwrite(img_path, face_region)
                            student_distraction_count[track_id]["image"] = img_path

                # Save JSON dynamically
                with open(DATA_FILE, "w") as f:
                    json.dump(student_distraction_count, f)

    out.write(frame)
    cv2.imshow("Classroom Distraction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()