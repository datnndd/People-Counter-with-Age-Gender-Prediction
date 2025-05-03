# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List
from typing import Any, List
import insightface
from insightface.app import FaceAnalysis
from shapely.geometry import Polygon, Point

# Region counting configuration
counting_regions = [
    {
        "name": "Region 1",
        "polygon": Polygon([(37, 880), (489, 744), (614, 761), (590, 914), (157, 957)]),
        "counts": 0,
        "region_color": (255, 42, 4),  # BGR
        "text_color": (255, 255, 255),  # White
    },
    {
        "name": "Region 2",
        "polygon": Polygon([(1691, 871), (1157, 1072), (377, 1069), (1234, 832)]),
        "counts": 0,
        "region_color": (37, 255, 225),  # BGR
        "text_color": (0, 0, 0),  # Black
    },
]

# def preprocess_frame(frame, denoise=True, sharpen=True):
#     """Tiền xử lý khung hình."""
#     # Denoise
#     denoised = cv2.fastNlMeansDenoisingColored(frame, None, h=3, templateWindowSize=7,
#                                                searchWindowSize=21) if denoise else frame
#
#     # Sharpen
#     if sharpen:
#         blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
#         sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
#     else:
#         sharpened = denoised
#
#     # CLAHE
#     lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l_clahe = clahe.apply(l)
#     merged = cv2.merge((l_clahe, a, b))
#     contrast_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
#
#     return contrast_enhanced

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
        # preprocess: bool = True,
        # denoise: bool = True,
        # sharpen: bool = True,
) -> None:
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

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

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        # Tiền xử lý frame
        # if preprocess:
        #     processed_frame = preprocess_frame(frame, denoise=denoise, sharpen=sharpen)
        # else:
        #     processed_frame = frame

        # Reset counts for each frame
        for region in counting_regions:
            region["counts"] = 0

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
                        region["counts"] += 1

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
    run(
        source='video_test/3105196-hd_1920_1080_30fps.mp4',
        view_img=True,
        save_img=True,
        device='cpu'
        # preprocess=False,
        # denoise=True,
        # sharpen=True
    )
