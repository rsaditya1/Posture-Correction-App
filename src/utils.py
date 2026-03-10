import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b given three 3D points a, b, c.
    Returns angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return angle


def calculate_inclination(a, b):
    """
    Calculate the inclination of line a->b relative to vertical.
    0 degrees = perfectly vertical.
    """
    a = np.array(a)
    b = np.array(b)

    diff = a - b
    vertical = np.array([0, -1, 0])

    cosine = np.dot(diff, vertical) / (np.linalg.norm(diff) * np.linalg.norm(vertical) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return angle


def extract_features_from_landmarks(landmarks_list):
    """
    Given a list of landmarks (each with x, y, z, visibility),
    extract posture-relevant features.
    Returns a dictionary or None if visibility too low.

    MediaPipe landmark indices:
        0: nose
        7: left ear
        8: right ear
        11: left shoulder
        12: right shoulder
        23: left hip
        24: right hip
    """

    def lm(idx):
        l = landmarks_list[idx]
        return [l.x, l.y, l.z]

    def vis(idx):
        return landmarks_list[idx].visibility

    key_indices = [0, 7, 8, 11, 12, 23, 24]
    min_visibility = min(vis(i) for i in key_indices)
    if min_visibility < 0.5:
        return None

    mid_shoulder = np.mean([lm(11), lm(12)], axis=0).tolist()

    features = {}

    features["neck_inclination_L"] = calculate_inclination(lm(7), lm(11))
    features["neck_inclination_R"] = calculate_inclination(lm(8), lm(12))

    features["torso_inclination_L"] = calculate_inclination(lm(11), lm(23))
    features["torso_inclination_R"] = calculate_inclination(lm(12), lm(24))

    features["head_angle_L"] = calculate_angle(lm(7), lm(11), lm(23))
    features["head_angle_R"] = calculate_angle(lm(8), lm(12), lm(24))

    features["shoulder_y_diff"] = abs(lm(11)[1] - lm(12)[1])

    features["head_forward_z"] = lm(0)[2] - mid_shoulder[2]

    features["nose_shoulder_y_diff"] = mid_shoulder[1] - lm(0)[1]

    features["neck_inclination_avg"] = (
        features["neck_inclination_L"] + features["neck_inclination_R"]
    ) / 2

    features["torso_inclination_avg"] = (
        features["torso_inclination_L"] + features["torso_inclination_R"]
    ) / 2

    return features