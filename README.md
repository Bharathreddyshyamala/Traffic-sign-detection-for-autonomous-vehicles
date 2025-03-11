# Traffic-sign-detection-for-autonomous-vehicles
**Introduction**
This project implements Traffic Sign Detection using YOLOv8, one of the most advanced object detection models available today. The primary objective is to accurately detect and classify various traffic signs from images or video streams, making it a crucial technology for applications such as:
* Enabling cars to automatically detect and react to traffic signs is known as autonomous driving.
* Traffic monitoring: Assisting law enforcement in effectively assessing and controlling traffic flow.
* Intelligent Transportation Systems (ITS): By using AI-powered detection, ITS helps create safer and smarter roads.

In order to control traffic, maintain road safety, and give drivers vital information, traffic signs are important.  However, it might be difficult to manually monitor and analyze them, particularly in dynamic areas like highways or metropolitan streets.  This research uses computer vision and deep learning to automate the procedure and provide very accurate real-time detection.

 A labeled dataset of traffic signs, comprising standard types like warning, prohibitory, stop, and speed restriction signs, is used to train the model.  The model is intended to provide high detection accuracy with minimal latency through extensive training, fine-tuning, and optimizations, which qualifies it for use in practical situations.

This project shows off YOLOv8's powerful and effective object identification features, which enable quick inference on a range of devices, including embedded AI systems, desktop GPUs, and edge devices.  This solution offers a solid basis for traffic sign identification tasks, whether for research, industry applications, or smart city efforts.

**Dataset**
The dataset used for training consists of images containing various traffic signs labeled with bounding boxes. The dataset includes:

🚏 Speed limit signs

⚠️ Warning signs

⛔ Prohibitory signs

🛑 Stop signs

📍 Directional signs

#Model Architecture
The project uses YOLOv8 (You Only Look Once, Version 8) for object detection, which provides:

✅ High-speed inference
✅ Optimized deep learning backbone
✅ Improved accuracy compared to previous YOLO versions
✅ Flexibility for real-time application


#Installation & Setup
To run this project on your local machine, follow these steps:

1️⃣ Clone the Repository
bash
'git clone https://github.com/your-repo/traffic-sign-detection-yolov8.git'
'cd traffic-sign-detection-yolov8'

2️⃣ Install Dependencies
Ensure you have Python installed, then install the required dependencies:

bash
'pip install ultralytics opencv-python numpy matplotlib'
Or install from requirements.txt:

bash
'pip install -r requirements.txt'

#3️⃣ Train the Model (Optional)
If you want to train YOLOv8 from scratch or fine-tune on a custom dataset
We trained the model using YOLOV8 BY BUILDING YAML

bash
'yolo task=detect mode=train model=yolov8s.pt data=dataset.yaml epochs=50 imgsz=640'
yolov8s.pt – Pretrained YOLOv8 model
dataset.yaml – Path to dataset configuration
epochs=50 – Number of training epochs
imgsz=640 – Image size for training


4️⃣ Run Inference
To test the trained model on images:

bash
yolo task=detect mode=predict model=best.pt source=sample_image.jpg
For real-time detection using a webcam:

bash
yolo task=detect mode=predict model=best.pt source=0


🖥 Results
📊 Model Performance Metrics:
mAP (Mean Average Precision): XX%
Inference Speed: XX FPS
Training Time: XX minutes

📌 Applications
🚗 Autonomous Driving Systems
🚦 Smart Traffic Monitoring
📊 Road Safety Analysis
🛣️ Intelligent Transportation Systems

🏆 Future Improvements
🔹 Fine-tune the model with more diverse datasets
🔹 Deploy as a real-time mobile/web application
🔹 Improve detection speed with hardware acceleration (TensorRT, OpenVINO)
🔹 Integrate with lane detection for better scene understanding

📝 References



✨ Contributors
👨‍💻 BHARATH REDDY SHYAMALA
📧 [your.email@example.com]

👨‍💻 HANEESH REDDY NEELA
📧 [your.email@example.com]

👨‍💻 KALYAN GUTTA
📧 [kalyangutta.18@gmail.com]
