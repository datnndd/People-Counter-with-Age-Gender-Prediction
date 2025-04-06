# 👁️‍🔮 People Counter with Age & Gender Prediction

Dự án AI giúc **đếm số lượng người** trong **khu vực khoanh vùng sẵn**, đồng thời **dự đoán tuổi và giới tính** của từng người, hiển thị trực tiếp trên màn hình giám sát.

## 📌 Mục tiêu
- 📍 Phát hiện người trong video/camera real-time.
- 🧠 Theo dõi người di chuyển trong khu vực đã định nghĩa (zone).
- 👨👩 Dự đoán tuổi và giới tính từ khuôn mặt người.
- 📺 Hiển thị số lượng người và thông tin lên màn hình.

## 📷 Demo
<img src="demo.gif" alt="demo" width="600"/>

*Ví dụ demo đang đếm người và dự đoán thông tin trong một khu vực cụ thể.*

## 🎯 Pipeline AI

📹 Video Input (.avi or .mp4)  
↓  
🧠 YOLO11n-pose → Person Detection, Tracking (ByteTrack)  
↓  
😶 Face Extraction  
↓  
🧓 Face Detection, Age & Gender Prediction (using InsightFace)  
↓  
📊 Hiển thị & đếm số người trong từng vùng.

## 🛠 Công nghệ sử dụng
- YOLO11n-pose – Phát hiện người, Theo dõi người
- InsightFace – Dự đoán tuổi, giới tính
- OpenCV – Hiển thị kết quả

## ⚙️ Cài đặt

```bash
git clone https://github.com/datnndd/People-Counter-with-Age-Gender-Prediction.git
cd People-Counter-with-Age-Gender-Prediction
pip install -r requirements.txt
```

## 🚀 Cách sử dụng

...

## 📊 Kết quả hiển thị
- Số lượng người trong vùng đã định sẵn
- Tuổi (ước lượng)
- Giới tính
- Bounding box + ID người + Overlay trực quan

## 📁 Các tính năng có thể mở rộng thêm
- Ghi file
- Cải thiện khả năng nhận diện đối tượng với SAHI

## 🤝 Đóng góp
Mọi ý kiến đóng góp đều rất được hoan nghênh! Bạn có thể tạo `Issue` hoặc gửi `Pull Request` nếu muốn cải thiện dự án.
