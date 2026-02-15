import numpy as np
from angle_utils import calculate_angle


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

def extract_features(sequence):
    """
    sequence: list of frames
    each frame: list of 25 (x,y,z)
    """
    features = []

    for frame in sequence:
        # 2D projection (x, y only)
        def p(j): 
            return frame[j][:2]

        left_elbow = calculate_angle(
            p(JOINTS["shoulder_left"]),
            p(JOINTS["elbow_left"]),
            p(JOINTS["wrist_left"])
        )

        right_elbow = calculate_angle(
            p(JOINTS["shoulder_right"]),
            p(JOINTS["elbow_right"]),
            p(JOINTS["wrist_right"])
        )

        left_knee = calculate_angle(
            p(JOINTS["hip_left"]),
            p(JOINTS["knee_left"]),
            p(JOINTS["ankle_left"])
        )

        right_knee = calculate_angle(
            p(JOINTS["hip_right"]),
            p(JOINTS["knee_right"]),
            p(JOINTS["ankle_right"])
        )

        torso_tilt = calculate_angle(
            p(JOINTS["spine_base"]),
            p(JOINTS["spine_mid"]),
            p(JOINTS["neck"])
        )

        features.append([
            left_elbow,
            right_elbow,
            left_knee,
            right_knee,
            torso_tilt
        ])

    return np.array(features)
