# AI Driver Safety Monitoring System (Kuppam)

A Python-based safety system built for **Python 3.12.6**.

### Features:
- **Drowsiness Detection**: 2-second buffer for eye closing.
- **Telugu Voice Alerts**: Safety warnings in local language.
- **Yawn Detection**: Suggests tea stalls in **Kuppam** after 3 yawns.
- **Seatbelt Interlock**: Engine only starts if seatbelt is detected.

### Setup:
1. Install requirements: `pip install opencv-python mediapipe flask numpy`
2. Download `face_landmarker.task` from Google MediaPipe and place in root.
3. Run `python app.py`.
