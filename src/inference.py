import os
import sys
import json
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import onnxruntime as ort

from utils import extract_features_from_landmarks


def run_inference(
    model_path="models/posture_model.onnx",
    feature_names_path="models/feature_names.json",
    pose_model_path="models/pose_landmarker.task",
    bad_posture_threshold=5,
):
    """Real-time posture monitoring with webcam."""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, model_path)
    feature_names_path = os.path.join(base_dir, feature_names_path)
    pose_model_path = os.path.join(base_dir, pose_model_path)

    # --- Load ONNX model ---
    print("Loading posture model...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(feature_names_path) as f:
        feature_names = json.load(f)

    print(f"Model loaded. Features: {len(feature_names)}")

    # --- Set up MediaPipe ---
    print("Loading pose landmarker...")
    base_options = mp_python.BaseOptions(
        model_asset_path=pose_model_path,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --- Open webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    # Get webcam resolution for overlay positioning
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {frame_width}x{frame_height}")

    print("\n" + "=" * 40)
    print("POSTURE MONITOR STARTED")
    print("Press 'q' to quit")
    print("=" * 40 + "\n")

    # Tracking variables
    bad_posture_start = None
    total_bad_seconds = 0
    total_good_seconds = 0
    session_start = time.time()
    frame_count = 0
    fps = 0
    fps_update_time = time.time()
    fps_frame_count = 0
    last_prediction = -1
    last_confidence = 0.0
    last_status_change = time.time()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break

            frame_count += 1
            current_time = time.time()

            # --- FPS calculation ---
            fps_frame_count += 1
            if current_time - fps_update_time >= 1.0:
                fps = fps_frame_count
                fps_frame_count = 0
                fps_update_time = current_time

            # --- MediaPipe pose detection ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((current_time - session_start) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            prediction = -1
            confidence = 0.0

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                # Extract features
                features = extract_features_from_landmarks(landmarks)

                if features is not None:
                    # Build input array in correct order
                    input_array = np.array(
                        [[features[name] for name in feature_names]],
                        dtype=np.float32,
                    )

                    # Run ONNX inference
                    outputs = session.run(None, {input_name: input_array})
                    prediction = int(outputs[0][0])

                    # Get confidence
                    if len(outputs) > 1:
                        probs = outputs[1]
                        if isinstance(probs, list):
                            confidence = float(probs[0].get(prediction, 1.0))
                        elif isinstance(probs, np.ndarray) and probs.ndim == 2:
                            confidence = float(probs[0][prediction])
                        else:
                            confidence = 1.0
                    else:
                        confidence = 1.0

                    last_prediction = prediction
                    last_confidence = confidence

                # Draw skeleton
                draw_skeleton(frame, landmarks)

            # --- Track posture duration ---
            if last_prediction == 0:
                if bad_posture_start is None:
                    bad_posture_start = current_time
                bad_duration = current_time - bad_posture_start
            else:
                if bad_posture_start is not None:
                    total_bad_seconds += current_time - bad_posture_start
                bad_posture_start = None
                bad_duration = 0

            # --- Draw overlay ---
            draw_overlay(
                frame,
                last_prediction,
                last_confidence,
                bad_duration,
                bad_posture_threshold,
                fps,
                current_time - session_start,
                total_bad_seconds + (bad_duration if last_prediction == 0 else 0),
            )

            cv2.imshow("Posture Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # --- Session summary ---
    session_duration = time.time() - session_start
    if bad_posture_start is not None:
        total_bad_seconds += time.time() - bad_posture_start

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print("SESSION SUMMARY")
    print("=" * 40)
    print(f"Duration:     {session_duration:.0f} seconds")
    print(f"Total frames: {frame_count}")
    print(f"Avg FPS:      {frame_count / session_duration:.1f}")
    print(f"Bad posture:  {total_bad_seconds:.0f}s ({total_bad_seconds/session_duration*100:.1f}%)")
    print(f"Good posture: {session_duration - total_bad_seconds:.0f}s ({(session_duration - total_bad_seconds)/session_duration*100:.1f}%)")
    print("=" * 40)


def draw_skeleton(frame, landmarks):
    """Draw pose landmarks and connections on frame."""

    h, w, _ = frame.shape

    connections = [
        (0, 7), (0, 8),
        (7, 11), (8, 12),
        (11, 12),
        (11, 13), (12, 14),
    ]

    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

    # Draw key landmarks with larger circles
    key_points = [0, 7, 8, 11, 12]
    for idx in key_points:
        if idx < len(landmarks):
            l = landmarks[idx]
            x, y = int(l.x * w), int(l.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)


def draw_overlay(frame, prediction, confidence, bad_duration, threshold, fps, elapsed, total_bad):
    """Draw status overlay on frame."""

    h, w, _ = frame.shape

    # --- Status bar background ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if prediction == 1:
        status = "GOOD POSTURE"
        color = (0, 220, 0)
        emoji = "[OK]"
    elif prediction == 0:
        status = "BAD POSTURE"
        color = (0, 0, 220)
        emoji = "[!!]"
    else:
        status = "NO POSE DETECTED"
        color = (128, 128, 128)
        emoji = "[??]"

    # Status text
    cv2.putText(
        frame, f"{emoji} {status}", (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
    )

    # Confidence
    cv2.putText(
        frame, f"Confidence: {confidence:.1%}", (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )

    # FPS and session time
    cv2.putText(
        frame, f"FPS: {fps}", (w - 120, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )
    cv2.putText(
        frame, f"Time: {int(elapsed)}s", (w - 120, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )

    # --- Bad posture alert ---
    if prediction == 0 and bad_duration >= threshold:
        # Flash effect
        flash = int(bad_duration * 2) % 2 == 0
        alert_color = (0, 0, 255) if flash else (0, 0, 180)

        # Red border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), alert_color, 6)

        # Alert banner at bottom
        banner_y = h - 80
        cv2.rectangle(frame, (0, banner_y), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"STRAIGHTEN UP! Bad posture for {bad_duration:.0f}s",
            (15, banner_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            alert_color,
            2,
        )

        # Progress bar showing how long in bad posture
        bar_width = min(int((bad_duration / 60) * (w - 30)), w - 30)
        cv2.rectangle(frame, (15, banner_y + 50), (15 + bar_width, banner_y + 60), alert_color, -1)
        cv2.rectangle(frame, (15, banner_y + 50), (w - 15, banner_y + 60), (100, 100, 100), 1)

    elif prediction == 0 and bad_duration > 0:
        # Warning: approaching threshold
        remaining = threshold - bad_duration
        cv2.putText(
            frame,
            f"Warning: {remaining:.0f}s until alert",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    # --- Session stats at bottom ---
    if prediction != 0 or bad_duration < threshold:
        stats_y = h - 25
        bad_pct = (total_bad / elapsed * 100) if elapsed > 0 else 0
        cv2.putText(
            frame,
            f"Session: {total_bad:.0f}s bad ({bad_pct:.0f}%)",
            (15, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time posture monitor")
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Seconds of bad posture before alert (default: 5)",
    )
    args = parser.parse_args()

    run_inference(bad_posture_threshold=args.threshold)