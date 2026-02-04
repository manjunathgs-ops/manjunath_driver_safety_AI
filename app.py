'''import cv2


import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify

# Setup for the new MediaPipe Tasks API (Python 3.12 compatible)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

app = Flask(__name__)

# Initialize the detector using the task file you just downloaded
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1)

detector = FaceLandmarker.create_from_options(options)

status = {"seatbelt": False, "drowsy": False}

# Correct landmarks for the Tasks model
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_ear(landmarks, eye_points):
    try:
        # The new API returns landmarks in a slightly different list format
        def get_pt(idx): return np.array([landmarks[idx].x, landmarks[idx].y])
        p1, p2, p3 = get_pt(eye_points[0]), get_pt(eye_points[1]), get_pt(eye_points[2])
        p4, p5, p6 = get_pt(eye_points[3]), get_pt(eye_points[4]), get_pt(eye_points[5])
        
        ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
        return ear
    except:
        return 0.3

def gen_frames():
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Convert frame to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Increment timestamp (required for Video Mode)
        frame_timestamp_ms += 33 
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            status["seatbelt"] = True 
            # result.face_landmarks is a list of lists; we take the first face [0]
            landmarks = result.face_landmarks[0]
            
            l_ear = get_ear(landmarks, LEFT_EYE)
            r_ear = get_ear(landmarks, RIGHT_EYE)
            avg_ear = (l_ear + r_ear) / 2.0
            
            # If eyes are closed for more than a few frames
            status["drowsy"] = avg_ear < 0.20
        else:
            status["seatbelt"] = False
            status["drowsy"] = False

        # Encode for Flask [OpenCV Docs]
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify(status)

if __name__ == '__main__':
    # Start the local server [Flask Quickstart](https://flask.palletsprojects.com)
    print("Project live at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)'''



'''import cv2

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# 1. Initialize AI Model (MediaPipe Tasks)
# Make sure 'face_landmarker.task' is in your D:\the_driver folder
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.4, # Lowered slightly to stop flickering
    min_face_presence_confidence=0.4
)
detector = vision.FaceLandmarker.create_from_options(options)

# Project State
status = {"seatbelt": False, "drowsy": False}

# Correct Eye Indices for the FaceLandmarker model
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def get_ear(landmarks, eye_points):
    try:
        # Landmarks come as a list of normalized keypoints
        def get_pt(idx): 
            return np.array([landmarks[idx].x, landmarks[idx].y])
            
        p1, p2, p3 = get_pt(eye_points[0]), get_pt(eye_points[1]), get_pt(eye_points[2])
        p4, p5, p6 = get_pt(eye_points[3]), get_pt(eye_points[4]), get_pt(eye_points[5])
        
        # EAR Formula calculation
        ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
        return ear
    except:
        return 0.3 # Default to "open" eye if calculation fails

def gen_frames():
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Performance: flip the image so it acts like a mirror
        frame = cv2.flip(frame, 1)
        
        # Prepare MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Timestamp must increase for Video Mode
        frame_timestamp_ms += 33 
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            status["seatbelt"] = True 
            # Get landmarks for the first face detected
            face_landmarks = result.face_landmarks[0]
            
            l_ear = get_ear(face_landmarks, LEFT_EYE_INDICES)
            r_ear = get_ear(face_landmarks, RIGHT_EYE_INDICES)
            avg_ear = (l_ear + r_ear) / 2.0
            
            # Use bool() to ensure JSON compatibility in Python 3.12
            status["drowsy"] = bool(avg_ear < 0.21)
        else:
            status["seatbelt"] = False
            status["drowsy"] = False

        # Encode frame for browser display
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    # Convert all values to standard Python types to avoid JSON errors
    clean_status = {k: bool(v) for k, v in status.items()}
    return jsonify(clean_status)

if __name__ == '__main__':
    # Start server [Flask Documentation](https://flask.palletsprojects.com)
    print("Project Running at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
'''


import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# --- AI Configuration ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Global State ---
status = {
    "seatbelt": False,
    "drowsy": False,
    "yawn_count": 0,
    "suggestion": "",
    "engine_started": False
}

eye_closed_start_time = None
yawn_active = False

# Landmark Indices for Eyes & Mouth [Google MediaPipe Docs]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Top and bottom inner lip points for yawning
UPPER_LIP = 13
LOWER_LIP = 14

def get_ear(landmarks, eye_points):
    try:
        def get_pt(idx): return np.array([landmarks[idx].x, landmarks[idx].y])
        p1, p2, p3, p4, p5, p6 = [get_pt(idx) for idx in eye_points]
        return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    except: return 0.3

def gen_frames():
    global eye_closed_start_time, yawn_active
    cap = cv2.VideoCapture(0)
    ts_ms = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts_ms += 33 
        result = detector.detect_for_video(mp_image, ts_ms)

        if result.face_landmarks:
            status["seatbelt"] = True 
            landmarks = result.face_landmarks[0]
            
            # 1. Eye Monitoring (Ignore < 2.0s)
            avg_ear = (get_ear(landmarks, LEFT_EYE) + get_ear(landmarks, RIGHT_EYE)) / 2.0
            if avg_ear < 0.20:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                elif time.time() - eye_closed_start_time > 2.0:
                    status["drowsy"] = True
            else:
                eye_closed_start_time = None
                status["drowsy"] = False

            # 2. Yawn Detection (Mouth Opening Ratio)
            lip_dist = np.linalg.norm(np.array([landmarks[UPPER_LIP].x, landmarks[UPPER_LIP].y]) - 
                                      np.array([landmarks[LOWER_LIP].x, landmarks[LOWER_LIP].y]))
            if lip_dist > 0.06: # Threshold for open mouth
                if not yawn_active:
                    status["yawn_count"] += 1
                    yawn_active = True
            else:
                yawn_active = False

            # 3. Cafe Suggestion
            if status["yawn_count"] >= 3:
                status["suggestion"] = "You've yawned 3+ times. Suggesting nearby cafes..."
        else:
            status.update({"seatbelt": False, "drowsy": False})

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/start_engine', methods=['POST'])
def start_engine():
    if status["seatbelt"]:
        status["engine_started"] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Buckle your seatbelt first!"})

@app.route('/get_status')
def get_status():
    return jsonify({k: (bool(v) if isinstance(v, bool) else v) for k, v in status.items()})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
