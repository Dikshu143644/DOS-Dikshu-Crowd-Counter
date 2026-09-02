"""
================================================================================
DOS · Dikshu AI Crowd Analytics - ByteTrack Engine
================================================================================
Author: Dikshu (DOS)
GitHub: https://github.com/Dikshu143644
License: MIT
Description:
    Live Crowd Counting & Multi-Object Tracking with YOLOv8 & ByteTrack
================================================================================
"""

import os
import argparse
import cv2
from ultralytics import YOLO
import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOS · Dikshu AI Vision Counter - ByteTrack Engine"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="lab.mp4",
        help="Input video file or webcam index ('0')"
    )
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int,
        help="Webcam resolution width and height"
    )
    parser.add_argument(
        "--model",
        default="yolo-head-detection.pt" if os.path.exists("yolo-head-detection.pt") else "yolov8s.pt",
        type=str,
        help="Model weights path or YOLO variant"
    )
    parser.add_argument(
        "--output",
        default="out.mp4",
        type=str,
        help="Output video file path"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO(args.model)
    box_annotator = sv.BoxAnnotator()
    tracker = sv.ByteTrack(frame_rate=30)

    ret, frame = cap.read()
    if not ret:
        print(f"[ERROR] Unable to read from video source: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap_out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame.shape[1], frame.shape[0])
    )

    all_ids = set()
    window_name = "DOS · Dikshu AI Vision Counter (ByteTrack)"

    print(f"[*] Starting ByteTrack Counter with model: {args.model}")
    print("[*] Press 'ESC' or 'q' to stop.")

    while ret:
        current_frame_ids = set()
        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[(detections.class_id == 0)]
        detections = tracker.update_with_detections(detections=detections)

        labels = []
        for _, _, confidence, class_id, tracker_id in detections:
            labels.append(f"ID #{tracker_id} ({confidence:0.2f})")
            all_ids.add(tracker_id)
            current_frame_ids.add(tracker_id)

        annotated_image = box_annotator.annotate(frame, detections=detections, labels=labels)

        # DOS HUD Display
        cv2.rectangle(annotated_image, (10, 10), (320, 75), (20, 20, 25), -1)
        cv2.rectangle(annotated_image, (10, 10), (320, 75), (0, 255, 200), 1)
        cv2.putText(
            annotated_image,
            "DOS · DIKSHU VISION (BYTETRACK)",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 200),
            1
        )
        cv2.putText(
            annotated_image,
            f"Active: {len(current_frame_ids):02d} | Total Unique: {len(all_ids):02d}",
            (18, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2
        )

        cv2.imshow(window_name, annotated_image)
        cap_out.write(annotated_image)

        key = cv2.waitKey(25) & 0xFF
        if key in [27, ord('q')]:
            break

        ret, frame = cap.read()

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()
    print("=" * 60)
    print(f"[*] ByteTrack Run Completed. Total Unique IDs Tracked: {len(all_ids)}")
    print(f"[*] Output saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
