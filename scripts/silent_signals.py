import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# ===================== CONFIG =====================
MODEL_PATH = "../models/silent_signals_intent.h5"
SEQUENCE_LENGTH = 30
FEATURES = 5

# Intent labels (binary model: 0 = no intent, 1 = intent)
INTENT_LABELS = ["NO ACTION", "ACTION STARTING"]

# ===================== LOAD MODEL =====================
model = load_model(MODEL_PATH)
print("✅ Intent model loaded")

# ===================== MEDIAPIPE =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# ===================== HELPERS =====================
def extract_features(landmarks):
    """Extract 5 posture features"""
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    shoulder_y = (l_sh.y + r_sh.y) / 2
    hip_y = (l_hip.y + r_hip.y) / 2
    torso_len = abs(shoulder_y - hip_y)

    return np.array([
        shoulder_y,
        hip_y,
        torso_len,
        nose.y,
        abs(l_sh.x - r_sh.x)
    ], dtype=np.float32)

def detect_posture(features):
    torso_len = features[2]
    return "SITTING" if torso_len < 0.20 else "STANDING"

def motion_energy(sequence):
    diffs = np.diff(sequence, axis=0)
    return np.mean(np.linalg.norm(diffs, axis=1))

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
print("🎥 Camera started")

sequence = deque(maxlen=SEQUENCE_LENGTH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    posture_text = "POSTURE: --"
    intent_text = "INTENT: --"
    confidence_text = ""

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        features = extract_features(results.pose_landmarks.landmark)
        sequence.append(features)

        posture = detect_posture(features)
        posture_text = f"POSTURE: {posture}"

        if len(sequence) == SEQUENCE_LENGTH:
            seq_array = np.array(sequence)
            motion = motion_energy(seq_array)

            input_data = seq_array.reshape(1, SEQUENCE_LENGTH, FEATURES)
            preds = model.predict(input_data, verbose=0)[0]
            confidence = float(np.max(preds))

            # 🔑 CORRECT THRESHOLDS (THIS WAS YOUR ISSUE)
            if motion < 0.02:
                intent_text = "INTENT: NO ACTION"
                confidence_text = ""
            elif motion < 0.08:
                intent_text = "INTENT: POSSIBLE INTENT"
                confidence_text = f"Confidence: {confidence:.2f}"
            else:
                intent_text = "INTENT: ACTION STARTING"
                confidence_text = f"Confidence: {confidence:.2f}"

    # ===================== DISPLAY =====================
    cv2.putText(frame, posture_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, intent_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if confidence_text:
        cv2.putText(frame, confidence_text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Silent Signals – Intent Prediction", frame)

    # ✅ QUIT WORKS PROPERLY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
