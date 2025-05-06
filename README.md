# 🚗 YOLOv8 ROI-Based Object Detection & Speed Estimation GUI

This project is a Python application that leverages the **YOLOv8** object detection model to perform **real-time detection, tracking, and speed estimation** of vehicles and pedestrians within a **user-defined Region of Interest (ROI)** in a video. A user-friendly **Tkinter GUI** allows intuitive video selection and processing.

---

## 🛠 Features

- ✅ ROI-based object detection using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- ✅ Real-time tracking of vehicles and pedestrians with persistent IDs
- ✅ Speed estimation (in km/h) for moving objects within the ROI
- ✅ On-screen display of bounding boxes, labels, and speeds
- ✅ Object counting per class (e.g., Car, Truck, Pedestrian)
- ✅ Tkinter-based GUI for file selection and control
- ✅ Smooth speed visualization with temporal averaging

---

## 📷 Supported Object Classes

This system tracks and estimates speed for the following YOLOv8 classes:

| Class ID | Class Name   |
|----------|--------------|
| 0        | Pedestrian   |
| 2        | Car          |
| 3        | Motorcycle   |
| 7        | Truck        |

---

## 🖥 GUI Overview

![GUI Screenshot Placeholder](#) *(Insert a screenshot here)*

1. **Select Video**: Opens file dialog to choose a video file.
2. **ROI Selection**: Uses OpenCV to select the region of interest.
3. **Start Processing**: Launches the detection and tracking window.

---
Here’s an additional section you can include in your README to showcase the screenshots for ROI selection and the application in action:

---

## 🖼 Screenshots

### 🎯 ROI Selection Interface

This window appears when the user selects a video and is prompted to define the Region of Interest (ROI). The ROI helps focus the detection and tracking on a specific part of the frame.

![ROI Selection Screenshot](screenshots/roi_selection.png)
*Draw a rectangle around the desired region to track vehicles/pedestrians.*

---

### 🚦 Real-Time Detection and Speed Estimation

This is the main output window showing the detected objects, assigned IDs, speed estimations (in km/h), and class labels within the ROI.

![Detection and Tracking Screenshot](screenshots/detection_tracking.png)
*Vehicles and pedestrians are tracked with real-time speed estimates and object IDs.*

## ⚙️ Installation

### 🔧 Requirements

- Python 3.10.11
- CUDA-enabled GPU recommended (for real-time performance)

### 📦 Dependencies

Install dependencies via pip:

```bash
pip install ultralytics==8.3.116
opencv-python==4.10.0.84
deep-sort-realtime==1.3.2
numpy==1.25.2
torch==2.5.1+cu118
torchvision==0.20.1+cu118
torchaudio==2.5.1+cu118
matplotlib==3.10.0
seaborn==0.13.2
tqdm==4.67.1
pandas==2.2.3
scipy==1.9.3

