# 👁️‍�️ People Counter with Age & Gender Prediction

An AI-powered application that counts people within specified zones and predicts their age and gender in real-time. This project leverages YOLO for detection/tracking and InsightFace for facial analysis.

## 📌 Features

*   **Real-time Person Detection & Tracking**: Uses YOLO11n-pose and ByteTrack for robust tracking.
*   **Zone-based Counting**: Define custom polygon zones to count people entering specific areas.
*   **Demographic Analysis**: Estimates age and gender using InsightFace.
*   **Visual Overlay**: Displays bounding boxes, IDs, keypoints, and demographic info directly on the video feed.

## 📷 Demo

> *Add your demo GIFs here. Example:*
>
> ![Demo](demo/test1.gif)

## 🛠 Tech Stack

*   **Detection/Tracking**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (Pose Estimation)
*   **Face Analysis**: [InsightFace](https://github.com/deepinsight/insightface)
*   **Visualization**: OpenCV
*   **Geometry**: Shapely (for polygon zone calculations)

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/datnndd/People-Counter-with-Age-Gender-Prediction.git
    cd People-Counter-with-Age-Gender-Prediction
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Weights**:
    Ensure you have the model weights (e.g., `yolo11n-pose.pt`) in the project directory.

## 🚀 Usage

### 1. Configure Counting Zones
Use [Roboflow PolygonZone](https://polygonzone.roboflow.com/) to define your zones:
1.  Upload a reference frame from your video.
2.  Draw your polygons.
3.  Copy the coordinates into `regions.json`.

*(See `regions.json` for the expected format)*

### 2. Run the Application

**Basic Usage (Video File):**
```bash
python main.py --source "video_test/test1.mp4"
```

**Using Webcam:**
```bash
python main.py --source 0
```

**Full Options:**
```bash
python main.py \
    --source "video_test/test1.mp4" \
    --regions_config "regions.json" \
    --device "cuda" \
    --view_img True \
    --save_img True
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--source` | Path to video file or webcam index (0, 1) | **Required** |
| `--weights` | Path to YOLO model weights | `yolo11n-pose.pt` |
| `--device` | Computing device (`cpu` or `cuda`) | `cpu` |
| `--regions_config` | Path to JSON file defining zones | `regions.json` |
| `--view_img` | Display the processing window | `True` |
| `--save_img` | Save the output video to `video_output/` | `False` |

## 🤝 Contribution
Contributions are welcome! Please feel free to verify functionality, submit issues, or create pull requests.

## 📜 References
*   [Ultralytics YOLOv8 Region Counter](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter)
*   [InsightFace Project](https://github.com/deepinsight/insightface)
