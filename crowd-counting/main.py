"""
================================================================================
DOS · Dikshu AI Crowd Analytics - BoTSORT Engine
================================================================================
Author: Dikshu (DOS)
GitHub: https://github.com/Dikshu143644
License: MIT
Description:
    Live Crowd Counting & Head Tracking with YOLOv8 & BoTSORT Tracker (OSNet Re-ID)
================================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort

frame_width, frame_height = 1280, 720
model_path = "./yolo-head-detection.pt" if os.path.exists("./yolo-head-detection.pt") else "yolov8s.pt"
fps = 25
model = YOLO(model_path, verbose=False)

def calculate_color_based_on_id(track_id: int):
    colors = [
        [0, 255, 255],  # Neon Yellow
        [255, 0, 128],  # DOS Magenta
        [0, 255, 0],    # Neon Green
        [255, 128, 0],  # Orange
        [0, 191, 255],  # Sky Blue
        [255, 0, 255],  # Fuchsia
        [128, 128, 128],# Gray
        [255, 255, 255],# White
        [0, 250, 154],  # Spring Green
        [192, 192, 192] # Silver
    ]
    return colors[track_id % len(colors)]


def main():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("lab.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    reid_weights = Path('osnet_x0_25_msmt17.pt')
    if reid_weights.exists():
        tracker = BotSort(
            reid_weights=reid_weights,
            half=False,
            device="cpu",
            fp16=False,
            track_buffer=fps * 15,
            frame_rate=fps,
            fuse_first_associate=True
        )
    else:
        tracker = BotSort(half=False, device="cpu", fp16=False, track_buffer=fps * 15, frame_rate=fps)

    thickness = 2
    count = 0
    id_set = set()
    window_name = 'DOS · Dikshu AI Vision Counter (BoTSORT)'

    print(f"[*] Starting DOS Vision Counter using model: {model_path}")
    print("[*] Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        detections = model.predict(frame, verbose=False)[0]

        try:
            boxes_data = np.array(detections.boxes.data.tolist())
            if boxes_data.size > 0:
                ts = tracker.update(boxes_data, frame)
                if ts.shape[0] != 0:
                    xyxys = ts[:, 0:4].astype('int')
                    ids = ts[:, 4].astype('int')
                    confs = ts[:, 5]
                    clss = ts[:, 6]

                    for xyxy, track_id, conf, cls in zip(xyxys, ids, confs, clss):
                        if cls != 0 or conf < 0.65:
                            continue
                        id_set.add(track_id)
                        color = calculate_color_based_on_id(track_id)
                        cv2.rectangle(
                            frame,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            color,
                            thickness
                        )
                        cv2.putText(
                            frame,
                            f'ID: {track_id}',
                            (xyxy[0], max(15, xyxy[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            thickness
                        )

            # DOS HUD Overlay
            cv2.rectangle(frame, (10, 10), (280, 85), (20, 20, 25), -1)
            cv2.rectangle(frame, (10, 10), (280, 85), (0, 255, 200), 1)
            cv2.putText(
                frame,
                "DOS · DIKSHU VISION",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 200),
                1
            )
            cv2.putText(
                frame,
                f'Total Unique Count: {len(id_set)}',
                (18, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f'Processed Frames : {count}',
                (18, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1
            )

            cv2.imshow(window_name, frame)

        except Exception as e:
            pass

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=" * 60)
    print(f"[*] DOS Vision Counter Finished. Total Unique Count: {len(id_set)}")
    print(f"[*] All Unique Track IDs: {sorted(list(id_set))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
