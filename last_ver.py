# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import json
from typing import List, Any
import insightface
from insightface.app import FaceAnalysis
from shapely.geometry import Polygon, Point

def load_regions_from_json(json_path: str) -> List[dict]:
    with open(json_path, 'r') as f:
        regions_data = json.load(f)
    regions = []
    for region in regions_data:
        polygon = Polygon(region["polygon"])
        regions.append({
            "name": region["name"],
            "polygon": polygon,
            "counts": 0,
            "region_color": tuple(region["region_color"]),
            "text_color": tuple(region["text_color"])
        })
    return regions


def run(
        weights: str = 'yolo11n-pose.pt',
        source: str = None,
        device: str = "cpu",
        view_img: bool = True,
        save_img: bool = False,
        exist_ok: bool = False,
        classes: List[int] = None,
        line_thickness: int = 2,
        region_thickness: int = 2,
        regions_config: str = None  # Đường dẫn đến file JSON
) -> None:
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    if regions_config:
        if not Path(regions_config).exists():
            raise FileNotFoundError(f"Regions config path '{regions_config}' does not exist.")

        # Kiểm tra file có nội dung
        if Path(regions_config).stat().st_size == 0:
            raise ValueError(f"Regions config file '{regions_config}' is empty.")

        # Kiểm tra nội dung JSON hợp lệ
        try:
            counting_regions = load_regions_from_json(regions_config)
            if not counting_regions:  # Kiểm tra list regions không rỗng
                raise ValueError("No valid regions found in the config file.")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{regions_config}'.")
        except Exception as e:
            raise ValueError(f"Error loading regions config: {str(e)}")

    # Initialize models
    model = YOLO(weights)
    model.to(device)

    # Khởi tạo model InsightFace
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    id_info = {}

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))

    if save_img:
        save_dir = Path("video_output/")
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.avi"),
                                       cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    region_tracked_ids = {region["name"]: set() for region in counting_regions}
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        # Run tracking trên frame đã tiền xử lý
        results = model.track(frame, persist=True, classes=classes, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            keypoint = results[0].keypoints.xy.cpu().numpy()
            confidences = results[0].keypoints.conf.cpu().numpy()

            for box, track_id, cls, kp, conf in zip(boxes, track_ids, clss, keypoint, confidences):
                x1, y1, x2, y2 = map(int, box[:4])
                bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Check if center point is inside any region
                for region in counting_regions:
                    if region["polygon"].contains(Point(bbox_center)):
                        if track_id not in region_tracked_ids[region["name"]]:
                            region["counts"] += 1
                            region_tracked_ids[region["name"]].add(track_id)

                # Original processing (pose, age, gender)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
                text = f'ID: {track_id}'

                # Head detection and age/gender estimation
                if len(conf) >= 7 and (conf[5] >= 0.3 and conf[6] >= 0.3):
                    left_shoulder = kp[5]
                    right_shoulder = kp[6]
                    mid_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
                    mid_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
                    D = mid_y - y1

                    if D > 0:
                        head_x1 = max(0, int(mid_x - D / 2))
                        head_y1 = max(0, y1)
                        head_x2 = min(frame_width, int(mid_x + D / 2))
                        head_y2 = min(frame_height, y1 + D)
                        cv2.rectangle(frame, (head_x1, head_y1), (head_x2, head_y2), (0, 0, 255), line_thickness)

                        if track_id not in id_info:
                            head_roi = frame[head_y1:head_y2, head_x1:head_x2]
                            if head_roi.size != 0:
                                faces = app.get(head_roi)
                                if len(faces) > 0:
                                    face = faces[0]
                                    age = int(face.age)
                                    gender = 'Male' if face.gender == 1 else 'Female'
                                    id_info[track_id] = (age, gender)

                # Update text if age/gender info available
                if track_id in id_info:
                    age, gender = id_info[track_id]
                    text = f'ID: {track_id}, Age: {age}, Gender: {gender}'

                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw regions and counts
        for region in counting_regions:
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid = [int(coord) for coord in region["polygon"].centroid.coords[0]]

            # Draw polygon
            cv2.polylines(frame, [polygon_coords], isClosed=True,
                          color=region["region_color"], thickness=region_thickness)

            # Draw count text
            cv2.putText(frame, str(region["counts"]), (centroid[0] - 10, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, region["text_color"], 2)

        # Display or save results
        if view_img:
            cv2.imshow('YOLO Pose Tracking with Region Counting', frame)

        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videocapture.release()
    if save_img:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Pose Tracking with Region Counting')
    parser.add_argument('--source', type=str, required=True, help='Đường dẫn đến video')
    parser.add_argument('--weights', type=str, default='yolo11n-pose.pt', help='Đường dẫn đến weights YOLO')
    parser.add_argument('--device', type=str, default='cpu', help='Thiết bị (cpu/cuda)')
    parser.add_argument('--view_img', type=bool, default=True, help='Hiển thị video')
    parser.add_argument('--save_img', type=bool, default=False, help='Lưu video kết quả')
    parser.add_argument('--regions_config', type=str, help='Đường dẫn đến file JSON cấu hình regions')
    args = parser.parse_args()

    run(
        source=args.source,
        weights=args.weights,
        device=args.device,
        view_img=args.view_img,
        save_img=args.save_img,
        regions_config=args.regions_config
    )