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

## ⚙️ Installation

### 🔧 Requirements

- Python 3.8+
- CUDA-enabled GPU recommended (for real-time performance)

### 📦 Dependencies

Install dependencies via pip:

```bash
pip install ultralytics opencv-python numpy torch tkinter
