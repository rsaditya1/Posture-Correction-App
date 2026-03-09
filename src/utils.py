import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b given three 3D points a, b, c.
    Returns angle in degrees.

    Example: to get the elbow angle, pass shoulder, elbow, wrist.
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
    Calculate the inclination angle of line segment a->b relative to the vertical axis.
    0 degrees = perfectly vertical (good posture).
    Returns angle in degrees.

    Example: to get neck inclination, pass ear, shoulder.
    """
    a = np.array(a)
    b = np.array(b)

    diff = a - b
    # Vertical reference vector (pointing up in MediaPipe coordinates)
    vertical = np.array([0, -1, 0])

    cosine = np.dot(diff, vertical) / (np.linalg.norm(diff) * np.linalg.norm(vertical) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return angle


def extract_features(landmarks):
    """
    Given MediaPipe pose landmarks, extract posture-relevant features.
    Returns a dictionary of feature names and values.

    MediaPipe landmark indices:
        0: nose
        7: left ear
        8: right ear
        11: left shoulder
        12: right shoulder
        23: left hip
        24: right hip
        13: left elbow
        14: right elbow
    """

    def lm(idx):
        """Extract [x, y, z] from a landmark by index."""
        return [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

    def vis(idx):
        """Get visibility score for a landmark."""
        return landmarks[idx].visibility

    # Check minimum visibility threshold
    key_indices = [0, 7, 8, 11, 12, 23, 24]
    min_visibility = min(vis(i) for i in key_indices)
    if min_visibility < 0.5:
        return None  # Landmarks not reliable enough

    # Midpoints
    mid_shoulder = np.mean([lm(11), lm(12)], axis=0).tolist()
    mid_hip = np.mean([lm(23), lm(24)], axis=0).tolist()
    mid_ear = np.mean([lm(7), lm(8)], axis=0).tolist()

    features = {}

    # 1. Neck inclination: ear to shoulder vs vertical
    features["neck_inclination_L"] = calculate_inclination(lm(7), lm(11))
    features["neck_inclination_R"] = calculate_inclination(lm(8), lm(12))

    # 2. Torso inclination: shoulder to hip vs vertical
    features["torso_inclination_L"] = calculate_inclination(lm(11), lm(23))
    features["torso_inclination_R"] = calculate_inclination(lm(12), lm(24))

    # 3. Shoulder-ear-hip angle (how far head is forward)
    features["head_angle_L"] = calculate_angle(lm(7), lm(11), lm(23))
    features["head_angle_R"] = calculate_angle(lm(8), lm(12), lm(24))

    # 4. Shoulder alignment: difference in y-coordinates
    # Large difference = one shoulder higher than the other
    features["shoulder_y_diff"] = abs(lm(11)[1] - lm(12)[1])

    # 5. Head forward displacement
    # How far nose is in front of shoulder midpoint (z-axis)
    features["head_forward_z"] = lm(0)[2] - mid_shoulder[2]

    # 6. Nose to shoulder midpoint vertical offset
    features["nose_shoulder_y_diff"] = mid_shoulder[1] - lm(0)[1]

    # 7. Ear to shoulder midpoint (averaged)
    features["neck_inclination_avg"] = (
        features["neck_inclination_L"] + features["neck_inclination_R"]
    ) / 2

    # 8. Torso inclination averaged
    features["torso_inclination_avg"] = (
        features["torso_inclination_L"] + features["torso_inclination_R"]
    ) / 2

    return features