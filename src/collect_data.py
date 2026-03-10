import csv
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
#from mediapipe import solutions

from utils import extract_features_from_landmarks


def download_model(model_path):
    """Download the pose landmarker model if not present."""
    if os.path.exists(model_path):
        return

    print(f"Downloading pose landmarker model...")
    import urllib.request

    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(url, model_path)
    print(f"Model saved to {model_path}")


def collect_data(label, output_dir="../data/raw", duration_seconds=120, fps_limit=15):
    """
    Collect posture data from webcam using MediaPipe Tasks API.

    Args:
        label: 1 for good posture, 0 for bad posture
        output_dir: where to save the CSV
        duration_seconds: how long to record
        fps_limit: max frames per second to capture
    """

    os.makedirs(output_dir, exist_ok=True)

    # Download model if needed
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(model_dir, "pose_landmarker.task")
    download_model(model_path)

    label_name = "good" if label == 1 else "bad"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"posture_{label_name}_{timestamp}.csv")

    # Set up MediaPipe Pose Landmarker (new Tasks API)
    base_options = mp_python.BaseOptions(
        model_asset_path=model_path,
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print(f"\nRecording {label_name.upper()} posture for {duration_seconds} seconds.")
    print("Press 'q' to stop early.")
    print("Get into position...")
    print("Starting in 3 seconds...")
    time.sleep(1)
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    print("GO!\n")

    row_count = 0
    start_time = time.time()
    min_frame_interval = 1.0 / fps_limit
    last_frame_time = 0
    frame_timestamp_ms = 0

    header_written = False
    csvfile = open(filename, "w", newline="")
    writer = None

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed >= duration_seconds:
                print(f"\nTime limit reached ({duration_seconds}s).")
                break

            # FPS limiting
            if current_time - last_frame_time < min_frame_interval:
                ret, frame = cap.read()
                continue

            last_frame_time = current_time

            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose (VIDEO mode requires increasing timestamps)
            frame_timestamp_ms = int((current_time - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                # Extract features
                features = extract_features_from_landmarks(landmarks)

                if features is not None:
                    features["label"] = label

                    if not header_written:
                        writer = csv.DictWriter(csvfile, fieldnames=features.keys())
                        writer.writeheader()
                        header_written = True

                    writer.writerow(features)
                    row_count += 1

                # Draw landmarks on frame for visual feedback
                draw_landmarks_on_frame(frame, landmarks)

            # Display info
            remaining = int(duration_seconds - elapsed)
            status_color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.putText(
                frame,
                f"Recording: {label_name.upper()} posture",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                frame,
                f"Rows: {row_count} | Time left: {remaining}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Posture Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped early by user.")
                break

    csvfile.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Saved {row_count} rows to {filename}")


def draw_landmarks_on_frame(frame, landmarks):
    """Draw pose landmarks on the frame for visual feedback."""
    h, w, _ = frame.shape

    # Connections for upper body (what we care about for posture)
    connections = [
        (0, 7), (0, 8),      # nose to ears
        (7, 11), (8, 12),    # ears to shoulders
        (11, 12),            # shoulder to shoulder
        (11, 23), (12, 24),  # shoulders to hips
        (23, 24),            # hip to hip
    ]

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect posture data from webcam")
    parser.add_argument(
        "--label",
        type=int,
        required=True,
        choices=[0, 1],
        help="0 = bad posture, 1 = good posture",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Recording duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Max frames per second (default: 15)",
    )

    args = parser.parse_args()
    collect_data(label=args.label, duration_seconds=args.duration, fps_limit=args.fps)