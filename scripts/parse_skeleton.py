import os
import numpy as np
from angle_utils import calculate_angle

# =========================
# CONFIG
# =========================
DATASET_PATH = "../data/ntu_skeletons"
WINDOW_SIZE = 30        # 1 second intent window
MAX_FILES = 10          # limit for testing (increase later)

# NTU joint indices (0-based)
JOINTS = {
    "spine_base": 0,
    "spine_mid": 1,
    "neck": 2,
    "head": 3,
    "shoulder_left": 4,
    "elbow_left": 5,
    "wrist_left": 6,
    "shoulder_right": 8,
    "elbow_right": 9,
    "wrist_right": 10,
    "hip_left": 12,
    "knee_left": 13,
    "ankle_left": 14,
    "hip_right": 16,
    "knee_right": 17,
    "ankle_right": 18,
}

# =========================
# PARSE SINGLE FILE
# =========================
def parse_skeleton_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    idx = 0
    num_frames = int(lines[idx].strip())
    idx += 1

    frames = []

    for _ in range(num_frames):
        num_bodies = int(lines[idx].strip())
        idx += 1

        if num_bodies == 0:
            continue

        # Take FIRST body only
        idx += 1  # skip body info line
        num_joints = int(lines[idx].strip())
        idx += 1

        joints = []
        for _ in range(num_joints):
            values = list(map(float, lines[idx].strip().split()))
            joints.append(values[:3])  # x, y, z
            idx += 1

        frames.append(joints)

    return np.array(frames)  # (frames, 25, 3)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(sequence):
    features = []

    for frame in sequence:
        try:
            left_elbow = calculate_angle(
                frame[JOINTS["shoulder_left"]],
                frame[JOINTS["elbow_left"]],
                frame[JOINTS["wrist_left"]],
            )

            right_elbow = calculate_angle(
                frame[JOINTS["shoulder_right"]],
                frame[JOINTS["elbow_right"]],
                frame[JOINTS["wrist_right"]],
            )

            left_knee = calculate_angle(
                frame[JOINTS["hip_left"]],
                frame[JOINTS["knee_left"]],
                frame[JOINTS["ankle_left"]],
            )

            right_knee = calculate_angle(
                frame[JOINTS["hip_right"]],
                frame[JOINTS["knee_right"]],
                frame[JOINTS["ankle_right"]],
            )

            spine_tilt = calculate_angle(
                frame[JOINTS["spine_base"]],
                frame[JOINTS["spine_mid"]],
                frame[JOINTS["neck"]],
            )

            features.append([
                left_elbow,
                right_elbow,
                left_knee,
                right_knee,
                spine_tilt
            ])
        except:
            continue

    return np.array(features)  # (frames, 5)

# =========================
# SLIDING WINDOWS
# =========================
def create_windows(features):
    windows = []
    for i in range(len(features) - WINDOW_SIZE):
        windows.append(features[i:i + WINDOW_SIZE])
    return np.array(windows)

# =========================
# LOAD DATASET
# =========================
def load_dataset(folder):
    sequences = []
    labels = []

    files = sorted(os.listdir(folder))[:MAX_FILES]

    for file in files:
        path = os.path.join(folder, file)
        frames = parse_skeleton_file(path)

        if len(frames) < WINDOW_SIZE:
            continue

        features = extract_features(frames)
        windows = create_windows(features)

        # Label rule (simple placeholder):
        # Even action IDs -> 0, Odd -> 1
        action_id = int(file.split("A")[-1].split(".")[0])
        label = action_id % 2

        for w in windows:
            sequences.append(w)
            labels.append(label)

    return np.array(sequences), np.array(labels)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X, y = load_dataset(DATASET_PATH)

    print("Loaded sequences:", len(X))
    print("Labels:", y[:5].tolist())
    print("Frames in one sample:", X.shape[1])
    print("Features per frame:", X.shape[2])
    print("First frame features:", np.round(X[0][0], 2))
