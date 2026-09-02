#!/usr/bin/env python3
"""
================================================================================
DOS · Dikshu AI Crowd Analytics & Vision Counter
================================================================================
Author: Dikshu (DOS)
GitHub: https://github.com/Dikshu143644
License: MIT
Description:
    Real-time Crowd Analytics, Head/Person Detection & Multi-Object Tracking (MOT)
    using YOLOv8, BoTSORT (with OSNet Re-ID), and ByteTrack.
================================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not found. Please install: pip install ultralytics")
    sys.exit(1)


# Preset Color Palette for Tracking IDs
PALETTE = [
    (0, 255, 255),    # Neon Yellow
    (255, 0, 128),    # DOS Magenta
    (0, 255, 0),      # Neon Green
    (255, 128, 0),    # Orange
    (0, 191, 255),    # Deep Sky Blue
    (255, 0, 255),    # Fuchsia
    (50, 205, 50),    # Lime
    (255, 215, 0),    # Gold
    (0, 250, 154),    # Medium Spring Green
    (147, 112, 219),  # Medium Purple
]


def get_color_for_id(track_id: int) -> tuple:
    """Returns a deterministic RGB/BGR color tuple for a given tracking ID."""
    return PALETTE[track_id % len(PALETTE)]


def draw_dos_hud(
    frame: np.ndarray,
    current_count: int,
    total_unique_count: int,
    fps: float,
    frame_idx: int,
    model_name: str,
    tracker_name: str
) -> np.ndarray:
    """
    Renders the modern DOS Dikshu Vision HUD overlay on top of the video frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent HUD Header Panel
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (10, 10), (min(w - 10, 480), 125), (18, 18, 24), -1)
    alpha = 0.75
    cv2.addWeighted(hud_bg, alpha, frame, 1 - alpha, 0, frame)

    # Outer border for HUD
    cv2.rectangle(frame, (10, 10), (min(w - 10, 480), 125), (0, 255, 200), 1)

    # Title
    cv2.putText(
        frame,
        "DOS · DIKSHU VISION HUD",
        (22, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (0, 255, 200),
        1,
        cv2.LINE_AA
    )

    # Subtitle / Telemetry Line
    cv2.putText(
        frame,
        f"ENGINE: {model_name} | TRACKER: {tracker_name.upper()}",
        (22, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # Live Active Count
    cv2.putText(
        frame,
        f"ACTIVE VISIBLE : {current_count:02d}",
        (22, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Total Cumulative Unique Visitors
    cv2.putText(
        frame,
        f"TOTAL COUNT    : {total_unique_count:02d}",
        (22, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Right side FPS / Frame Indicator
    fps_text = f"FPS: {fps:4.1f} | FRM: {frame_idx:05d}"
    cv2.putText(
        frame,
        fps_text,
        (min(w - 10, 480) - 175, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="DOS · Dikshu AI Crowd Analytics & Vision Counter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: Webcam index (e.g. 0, 1) or path to video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="crowd-counting/yolo-head-detection.pt",
        help="Path to YOLO weights (.pt) or standard model name (e.g. yolov8s.pt)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracking algorithm to use: 'botsort' (with OSNet) or 'bytetrack'"
    )
    parser.add_argument(
        "--reid-weights",
        type=str,
        default="crowd-counting/osnet_x0_25_msmt17.pt",
        help="Path to OSNet Re-ID weights (required for BoTSORT)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.60,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Frame width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Frame height"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video to crowd-counting/output_videos/"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("⚡ DOS · Dikshu AI Crowd Analytics & Vision Counter ⚡")
    print("=" * 70)

    # Determine input source (webcam index vs video file)
    if args.source.isdigit():
        source = int(args.source)
        source_name = f"Webcam_{source}"
    else:
        source = args.source
        source_name = Path(source).stem
        if not os.path.exists(source):
            print(f"[ERROR] Input video path not found: {source}")
            sys.exit(1)

    print(f"[*] Loading YOLO Model: {args.model}")
    # Fallback to yolov8s.pt if specialized weights not found locally
    if not os.path.exists(args.model) and not args.model.startswith("yolov8"):
        print(f"[!] Warning: Model weights '{args.model}' not found. Falling back to 'yolov8s.pt'.")
        model = YOLO("yolov8s.pt", verbose=False)
        model_display = "yolov8s.pt"
    else:
        model = YOLO(args.model, verbose=False)
        model_display = Path(args.model).name

    # Initialize Tracker
    tracker = None
    if args.tracker == "botsort":
        try:
            from boxmot import BotSort
            reid_path = Path(args.reid_weights)
            if not reid_path.exists():
                print(f"[!] Warning: ReID weights '{args.reid_weights}' not found. Initializing without ReID.")
                tracker = BotSort(half=False, device="cpu", fp16=False, track_buffer=45, frame_rate=25)
            else:
                tracker = BotSort(
                    reid_weights=reid_path,
                    half=False,
                    device="cpu",
                    fp16=False,
                    track_buffer=45,
                    frame_rate=25,
                    fuse_first_associate=True
                )
            print("[*] BoTSORT Tracker successfully initialized.")
        except Exception as e:
            print(f"[!] Could not initialize BoTSORT ({e}). Falling back to ByteTrack.")
            args.tracker = "bytetrack"

    if args.tracker == "bytetrack" and tracker is None:
        try:
            import supervision as sv
            tracker = sv.ByteTrack(frame_rate=30)
            box_annotator = sv.BoxAnnotator()
            print("[*] ByteTrack Tracker successfully initialized.")
        except Exception as e:
            print(f"[!] supervision / ByteTrack import error ({e}).")

    # Open Capture
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height

    # Setup Video Writer if requested
    cap_out = None
    if args.save:
        out_dir = Path("crowd-counting/output_videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"dos_counter_{source_name}_{args.tracker}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap_out = cv2.VideoWriter(str(out_path), fourcc, fps_input, (frame_w, frame_h))
        print(f"[*] Saving processed video to: {out_path}")

    cumulative_ids = set()
    frame_idx = 0
    prev_time = time.time()
    fps = fps_input

    window_name = "DOS · Dikshu AI Vision Counter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, frame_w), min(720, frame_h))

    print("[*] Starting processing loop. Press 'q' or 'ESC' in the display window to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_visible_ids = set()

        # Inference
        detections = model.predict(frame, verbose=False, conf=args.conf)[0]

        # Tracker logic
        if args.tracker == "botsort" and tracker is not None:
            try:
                boxes_data = np.array(detections.boxes.data.tolist())
                if boxes_data.size > 0:
                    ts = tracker.update(boxes_data, frame)
                    if ts.shape[0] != 0:
                        xyxys = ts[:, 0:4].astype("int")
                        ids = ts[:, 4].astype("int")
                        confs = ts[:, 5]
                        clss = ts[:, 6]

                        for xyxy, track_id, conf, cls in zip(xyxys, ids, confs, clss):
                            if cls != 0 or conf < args.conf:
                                continue

                            cumulative_ids.add(track_id)
                            current_visible_ids.add(track_id)
                            color = get_color_for_id(track_id)

                            # Bounding Box
                            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                            # Label
                            label = f"ID: {track_id}"
                            cv2.putText(
                                frame,
                                label,
                                (xyxy[0], max(15, xyxy[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                                cv2.LINE_AA
                            )
            except Exception as e:
                pass

        elif args.tracker == "bytetrack" and tracker is not None:
            try:
                import supervision as sv
                dets = sv.Detections.from_ultralytics(detections)
                dets = dets[(dets.class_id == 0) & (dets.confidence >= args.conf)]
                dets = tracker.update_with_detections(detections=dets)
                for _, _, confidence, class_id, tracker_id in dets:
                    cumulative_ids.add(tracker_id)
                    current_visible_ids.add(tracker_id)
                labels = [f"ID #{tid} ({conf:0.2f})" for _, _, conf, _, tid in dets]
                frame = box_annotator.annotate(frame, detections=dets, labels=labels)
            except Exception as e:
                pass
        else:
            # Raw YOLO Detections fallback without tracker
            for box in detections.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= args.conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    current_visible_ids.add(len(current_visible_ids) + 1)

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else fps
        prev_time = curr_time

        # Render HUD
        frame = draw_dos_hud(
            frame=frame,
            current_count=len(current_visible_ids),
            total_unique_count=len(cumulative_ids) if cumulative_ids else len(current_visible_ids),
            fps=fps,
            frame_idx=frame_idx,
            model_name=model_display,
            tracker_name=args.tracker
        )

        if cap_out is not None:
            cap_out.write(frame)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:  # 'q' or ESC
            print("[*] User interrupted.")
            break

    # Cleanup
    cap.release()
    if cap_out is not None:
        cap_out.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("📊 DOS AI VISION SESSION SUMMARY")
    print(f"Total Processed Frames : {frame_idx}")
    print(f"Total Unique People/IDs: {len(cumulative_ids)}")
    print(f"Tracked Unique IDs     : {sorted(list(cumulative_ids)) if cumulative_ids else 'N/A'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
