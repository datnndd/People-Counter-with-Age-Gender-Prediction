# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Any, List
import insightface
from insightface.app import FaceAnalysis


def run(
        weights: str = 'yolo11n-pose.pt',
        source: str = None,
        device: str = "cpu",
        view_img: bool = True,
        save_img: bool = False,
        exist_ok: bool = False,
        classes: List[int] = None,
        line_thickness: int = 2,
) -> None:
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Khởi tạo model YOLO
    model = YOLO(weights)
    model.to(device)

    # Khởi tạo model InsightFace
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))  # Sử dụng GPU: ctx_id=0, CPU: ctx_id=-1

    videocapture = cv2.VideoCapture(source)
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))

    if save_img:
        save_dir = Path("ultralytics_pose_output")
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.avi"),
                                       cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        results = model.track(frame, persist=True, classes=classes, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confidences = results[0].keypoints.conf.cpu().numpy()

            for box, track_id, cls, kp, conf in zip(boxes, track_ids, clss, keypoints, confidences):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Xử lý keypoints đầu
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

                        # Cắt vùng đầu và nhận diện
                        head_roi = frame[head_y1:head_y2, head_x1:head_x2]
                        if head_roi.size != 0:
                            faces = app.get(head_roi)
                            if len(faces) > 0:
                                face = faces[0]
                                age = int(face.age)
                                gender = 'Male' if face.gender == 1 else 'Female'
                                cv2.putText(frame, f'Age: {age}, Gender: {gender}',
                                            (head_x1, head_y2 + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if view_img:
            cv2.imshow('YOLO11 Pose Tracking', frame)

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
        source='video_input/12208359_1920_1080_60fps.mp4',
        view_img=True,
        save_img=False,
        device='cpu'
    )