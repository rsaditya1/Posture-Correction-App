import csv
import os
import time

import cv2
import mediapipe as mp

from utils import extract_features


def collect_data(label, output_dir="data/raw", duration_seconds=120, fps_limit=15):
    """
    Collect posture data from webcam.

    Args:
        label: 1 for good posture, 0 for bad posture
        output_dir: where to save the CSV
        duration_seconds: how long to record (default 2 minutes)
        fps_limit: max frames per second to capture
    """

    os.makedirs(output_dir, exist_ok=True)

    label_name = "good" if label == 1 else "bad"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/posture_{label_name}_{timestamp}.csv"

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print(f"Recording {label_name} posture for {duration_seconds} seconds...")
    print("Press 'q' to stop early.")
    print("Get into position. Starting in 3 seconds...")
    time.sleep(3)

    row_count = 0
    start_time = time.time()
    min_frame_interval = 1.0 / fps_limit

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        # Prepare CSV
        header_written = False
        csvfile = open(filename, "w", newline="")
        writer = None

        last_frame_time = 0

        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            # Check time limit
            if elapsed >= duration_seconds:
                print(f"\nTime limit reached ({duration_seconds}s).")
                break

            # FPS limiting
            if current_time - last_frame_time < min_frame_interval:
                ret, frame = cap.read()  # Still read to clear buffer
                continue

            last_frame_time = current_time

            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract features
                features = extract_features(landmarks)

                if features is not None:
                    features["label"] = label

                    # Write header on first valid row
                    if not header_written:
                        writer = csv.DictWriter(csvfile, fieldnames=features.keys())
                        writer.writeheader()
                        header_written = True

                    writer.writerow(features)
                    row_count += 1

                # Draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            # Display info on frame
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