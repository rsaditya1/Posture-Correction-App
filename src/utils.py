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


def calculate_inclination_2d(a, b):
    """
    Calculate the inclination of line a->b relative to vertical using x,y only.
    0 degrees = perfectly vertical.
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    # Angle relative to vertical (y-axis)
    angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-8))

    return angle


def extract_features_from_landmarks(landmarks_list):
    """
    Extract posture features using ONLY upper body landmarks.
    Works with laptop webcam where hips may not be visible.

    Landmarks used:
        0: nose
        2: left eye inner
        5: right eye inner
        7: left ear
        8: right ear
        9: mouth left
        10: mouth right
        11: left shoulder
        12: right shoulder
        13: left elbow
        14: right elbow
    """

    def lm(idx):
        l = landmarks_list[idx]
        return [l.x, l.y, l.z]

    def vis(idx):
        return landmarks_list[idx].visibility

    # Only check upper body landmarks
    key_indices = [0, 7, 8, 11, 12]
    min_visibility = min(vis(i) for i in key_indices)
    if min_visibility < 0.5:
        return None

    features = {}

    # --- Neck inclination ---
    # Ear to shoulder angle relative to vertical
    # Larger angle = head tilted or pushed forward
    features["neck_incl_L"] = calculate_inclination_2d(lm(7), lm(11))
    features["neck_incl_R"] = calculate_inclination_2d(lm(8), lm(12))
    features["neck_incl_avg"] = (features["neck_incl_L"] + features["neck_incl_R"]) / 2

    # --- Head forward position ---
    # Nose z relative to shoulder midpoint z
    # More negative = head further forward (bad)
    mid_shoulder_z = (lm(11)[2] + lm(12)[2]) / 2
    features["head_forward_z"] = lm(0)[2] - mid_shoulder_z

    # --- Nose to shoulder vertical distance ---
    # How high the nose is above shoulder midpoint
    mid_shoulder_y = (lm(11)[1] + lm(12)[1]) / 2
    features["nose_above_shoulder"] = mid_shoulder_y - lm(0)[1]

    # --- Shoulder alignment ---
    # Difference in y between left and right shoulder
    # Large value = uneven shoulders
    features["shoulder_y_diff"] = abs(lm(11)[1] - lm(12)[1])

    # --- Shoulder width (x distance) ---
    # Can indicate slouching if shoulders come forward and width decreases
    features["shoulder_width"] = abs(lm(11)[0] - lm(12)[0])

    # --- Ear alignment ---
    # Difference in y between ears — head tilt
    features["ear_y_diff"] = abs(lm(7)[1] - lm(8)[1])

    # --- Ear to shoulder vertical ratio ---
    # Ratio of ear-shoulder distance to shoulder width
    # Normalizes for distance from camera
    ear_shoulder_dist_L = np.sqrt(
        (lm(7)[0] - lm(11)[0])**2 + (lm(7)[1] - lm(11)[1])**2
    )
    ear_shoulder_dist_R = np.sqrt(
        (lm(8)[0] - lm(12)[0])**2 + (lm(8)[1] - lm(12)[1])**2
    )
    features["ear_shoulder_ratio_L"] = ear_shoulder_dist_L / (features["shoulder_width"] + 1e-8)
    features["ear_shoulder_ratio_R"] = ear_shoulder_dist_R / (features["shoulder_width"] + 1e-8)

    # --- Nose to ear angle (head droop) ---
    # Angle at ear between nose and shoulder
    # Smaller angle = head drooping forward
    features["head_droop_L"] = calculate_angle(lm(0), lm(7), lm(11))
    features["head_droop_R"] = calculate_angle(lm(0), lm(8), lm(12))

    # --- Eye to ear vertical (head tilt forward) ---
    # If eyes are much lower than ears, head is tilted forward
    mid_eye_y = (lm(2)[1] + lm(5)[1]) / 2
    mid_ear_y = (lm(7)[1] + lm(8)[1]) / 2
    features["eye_ear_y_diff"] = mid_eye_y - mid_ear_y

    return features