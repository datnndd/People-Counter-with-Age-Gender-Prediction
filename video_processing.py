import cv2
import numpy as np


def preprocess_frame(frame, target_width=640, denoise=True, sharpen=True):
    """Tiền xử lý khung hình."""
    # Resize
    h, w = frame.shape[:2]
    new_height = int(h * (target_width / w))
    resized = cv2.resize(frame, (target_width, new_height))

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, h=3, templateWindowSize=7,
                                               searchWindowSize=21) if denoise else resized

    # Sharpen
    if sharpen:
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
    else:
        sharpened = denoised

    # CLAHE
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    contrast_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return contrast_enhanced


video_path = "video_input/17526711-hd_2048_1080_30fps.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame)
    cv2.imshow('Processed Video', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()