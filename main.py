# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import json
from typing import List, Dict, Set, Tuple, Any, Optional
import insightface
from insightface.app import FaceAnalysis
from shapely.geometry import Polygon, Point

# Constants
TRACKER_CONFIG = "bytetrack.yaml"
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_WEIGHTS = 'yolo11n-pose.pt'

class RegionCounter:
    """
    Handles region-based counting and drawing.
    """
    def __init__(self, config_path: str):
        self.regions = self._load_regions(config_path)
        self.tracked_ids: Dict[str, Set[int]] = {region["name"]: set() for region in self.regions}

    def _load_regions(self, json_path: str) -> List[dict]:
        """Loads and parses region configuration from a JSON file."""
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Regions config path '{json_path}' does not exist.")
        
        try:
            with open(json_path, 'r') as f:
                regions_data = json.load(f)
            
            if not regions_data:
                raise ValueError("No valid regions found in the config file.")

            parsed_regions = []
            for region in regions_data:
                polygon = Polygon(region["polygon"])
                parsed_regions.append({
                    "name": region["name"],
                    "polygon": polygon,
                    "counts": 0,
                    "region_color": tuple(region["region_color"]),
                    "text_color": tuple(region["text_color"])
                })
            return parsed_regions
        except Exception as e:
            raise ValueError(f"Error loading regions config: {str(e)}")

    def update_counts(self, bbox_center: Tuple[int, int], track_id: int):
        """Checks if a point is within any region and updates counts."""
        point = Point(bbox_center)
        for region in self.regions:
            if region["polygon"].contains(point):
                if track_id not in self.tracked_ids[region["name"]]:
                    region["counts"] += 1
                    self.tracked_ids[region["name"]].add(track_id)

    def draw_regions(self, frame: np.ndarray, thickness: int = 2):
        """Draws regions and their counts on the frame."""
        for region in self.regions:
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid = [int(coord) for coord in region["polygon"].centroid.coords[0]]

            # Draw polygon
            cv2.polylines(frame, [polygon_coords], isClosed=True,
                          color=region["region_color"], thickness=thickness)

            # Draw count text
            cv2.putText(frame, str(region["counts"]), (centroid[0] - 10, centroid[1]),
                        FONT, 0.7, region["text_color"], 2)


class PeopleAnalyzer:
    """
    Handles Face Analysis (Age/Gender) using InsightFace.
    """
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.id_info: Dict[int, Tuple[int, str]] = {}

    def analyze_face(self, frame: np.ndarray, head_box: Tuple[int, int, int, int], track_id: int):
        """
        Analyzes the face within the given bounding box if not already analyzed.
        """
        if track_id in self.id_info:
            return

        x1, y1, x2, y2 = head_box
        head_roi = frame[y1:y2, x1:x2]
        
        if head_roi.size == 0:
            return

        faces = self.app.get(head_roi)
        if len(faces) > 0:
            face = faces[0]
            age = int(face.age)
            gender = 'Male' if face.gender == 1 else 'Female'
            self.id_info[track_id] = (age, gender)

    def get_info(self, track_id: int) -> Optional[str]:
        """Returns formatted string of age/gender if available."""
        if track_id in self.id_info:
            age, gender = self.id_info[track_id]
            return f'Age: {age}, Gender: {gender}'
        return None

def initialize_video_writer(source: str, fps: int, size: Tuple[int, int], save_dir: str = "video_output") -> cv2.VideoWriter:
    """Initializes the VideoWriter."""
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(source).stem}.avi"
    return cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

def process_video(
        source: str,
        weights: str,
        device: str,
        view_img: bool,
        save_img: bool,
        regions_config: str
) -> None:
    
    if not Path(source).exists() and source != "0" and source != "1": # handling webcam indices roughly
        # Note: simplistic check for webcam indices passed as strings
         if not (source.isdigit() and int(source) in [0, 1]):
             raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Initialize Models
    model = YOLO(weights)
    model.to(device)
    
    analyzer = PeopleAnalyzer()
    
    # Load Regions if config provided
    counter = None
    if regions_config:
        counter = RegionCounter(regions_config)

    # Video Setup
    # Handle int source for webcams
    cap_source = int(source) if source.isdigit() else source
    videocapture = cv2.VideoCapture(cap_source)
    
    if not videocapture.isOpened():
         raise ValueError(f"Could not open video source: {source}")

    frame_width = int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(videocapture.get(cv2.CAP_PROP_FPS)) or 30 # Default to 30 if cannot read

    video_writer = None
    if save_img:
        video_writer = initialize_video_writer(str(source) if not isinstance(source, int) else "webcam", fps, (frame_width, frame_height))

    print(f"Processing video: {source}...")
    
    try:
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                break

            # Tracking
            results = model.track(frame, persist=True, tracker=TRACKER_CONFIG, verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy()
                confidences = results[0].keypoints.conf.cpu().numpy()

                for box, track_id, kp, conf in zip(boxes, track_ids, keypoints, confidences):
                    x1, y1, x2, y2 = map(int, box[:4])
                    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Update Counts
                    if counter:
                        counter.update_counts(bbox_center, track_id)

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare ID Text
                    id_text = f'ID: {track_id}'

                    # Estimate Head Position & Analyze Face
                    # Check for shoulders (indices 5 and 6) confidence
                    if len(conf) >= 7 and (conf[5] >= 0.3 and conf[6] >= 0.3):
                        left_shoulder = kp[5]
                        right_shoulder = kp[6]
                        mid_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
                        mid_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
                        
                        # Heuristic: estimation of head height based on shoulder-to-topbox distance
                        # Original logic: D = mid_y - y1.
                        # This assumes the person is upright and the box top is near the head top.
                        head_projection_height = mid_y - y1

                        if head_projection_height > 0:
                            head_x1 = max(0, int(mid_x - head_projection_height / 2))
                            head_y1 = max(0, y1)
                            head_x2 = min(frame_width, int(mid_x + head_projection_height / 2))
                            head_y2 = min(frame_height, y1 + head_projection_height)
                            
                            # Draw Head Box (Visual debugging)
                            cv2.rectangle(frame, (head_x1, head_y1), (head_x2, head_y2), (0, 0, 255), 2)

                            # Analyze Face
                            analyzer.analyze_face(frame, (head_x1, head_y1, head_x2, head_y2), track_id)

                    # Add Age/Gender info if available
                    info = analyzer.get_info(track_id)
                    if info:
                        id_text += f", {info}"
                    
                    cv2.putText(frame, id_text, (x1, y1 - 10), FONT, 0.6, (0, 255, 0), 2)

            # Draw Regions
            if counter:
                counter.draw_regions(frame)

            # Output
            if view_img:
                cv2.imshow('YOLO Pose Tracking & Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if save_img and video_writer:
                video_writer.write(frame)

    except KeyboardInterrupt:
        print("\nStopping processing...")
    finally:
        videocapture.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='People Counter with Age & Gender Prediction')
    parser.add_argument('--source', type=str, required=True, help='Video file path or camera ID (0, 1)')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS, help='YOLO model weights path')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu/cuda)')
    parser.add_argument('--regions_config', type=str, default='regions.json', help='Path to regions JSON config')
    parser.add_argument('--view_img', type=bool, default=True, help='Display video during processing')
    parser.add_argument('--save_img', type=bool, default=False, help='Save processed video to file')
    
    args = parser.parse_args()

    process_video(
        source=args.source,
        weights=args.weights,
        device=args.device,
        view_img=args.view_img,
        save_img=args.save_img,
        regions_config=args.regions_config
    )