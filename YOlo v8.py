import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import font as tkfont

# Load YOLOv8 model and move it to GPU
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('yolov8m.pt').to(device)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load YOLO model:\n{e}")
    exit()

# Confirmed Target Classes
TARGET_CLASSES = {
    2: 'Car',
    3: 'Motorcycle',
    7: 'Truck',
    0: 'Pedestrian'
}

# Tracking and Speed Estimation Parameters
tracker = {}
object_counter = defaultdict(int)
object_speed = {}
history = defaultdict(list)

DISTANCE_THRESHOLD = 80  # pixels
PIXELS_PER_FOOT_X = 8
PIXELS_PER_FOOT_Y = 20
SMOOTHING_WINDOW = 5

# Functions
def estimate_speed(prev_center, curr_center, fps):
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    pixel_distance = math.sqrt(dx**2 + dy**2)
    feet_distance = (pixel_distance / PIXELS_PER_FOOT_X)
    speed_fps = feet_distance * fps
    speed_kmph = speed_fps * 3.6
    return speed_kmph

def smooth_speed(id_, speed):
    history[id_].append(speed)
    if len(history[id_]) > SMOOTHING_WINDOW:
        history[id_].pop(0)
    return np.mean(history[id_])

def is_inside_roi(center, roi):
    x, y, w, h = roi
    return x <= center[0] <= x + w and y <= center[1] <= y + h

# Main Processing Function
def process_video(video_path, roi):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Unable to open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default fallback if fps not detected

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model.track(frame, persist=True, classes=list(TARGET_CLASSES.keys()))
            except Exception as e:
                messagebox.showerror("Error", f"YOLOv8 failed during tracking:\n{e}")
                break

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()

                for id_, box, cls in zip(ids, boxes, clss):
                    x1, y1, x2, y2 = box
                    class_name = TARGET_CLASSES.get(int(cls), None)
                    if class_name is None:
                        continue

                    center = ((x1 + x2) / 2, (y1 + y2) / 2)

                    if not is_inside_roi(center, roi):
                        continue

                    if id_ not in tracker:
                        tracker[id_] = {
                            "center": center,
                            "class_name": class_name,
                            "prev_center": center,
                            "speed": 0
                        }
                        object_counter[class_name] += 1

                    if id_ in tracker:
                        tracker[id_]["prev_center"] = tracker[id_]["center"]
                        tracker[id_]["center"] = center
                        speed = estimate_speed(tracker[id_]["prev_center"], center, fps)
                        smoothed_speed = smooth_speed(id_, speed)
                        tracker[id_]["speed"] = smoothed_speed

                    speed = tracker[id_]["speed"]

                    color = (0, 255, 0) if speed <= 30 else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name} {speed:.1f} km/h"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw ROI Rectangle
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display counts
            y_offset = 30
            for idx, (classname, count) in enumerate(object_counter.items()):
                text = f"{classname}: {count}"
                cv2.putText(frame, text, (10, y_offset + idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('YOLOv8 Detection in ROI', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        messagebox.showerror("Processing Error", f"An error occurred while processing the video:\n{e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# GUI
def select_video():
    try:
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Failed to read video. Try selecting another file.")
            return

        roi = cv2.selectROI("Select ROI and press ENTER", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        if roi == (0,0,0,0):
            messagebox.showerror("Error", "No ROI selected. Please try again.")
            return

        start_button.config(state=tk.NORMAL)
        start_button.video_path = video_path
        start_button.roi = roi

    except Exception as e:
        messagebox.showerror("Error", f"Failed to select video or ROI:\n{e}")

def start_processing():
    try:
        video_path = start_button.video_path
        roi = start_button.roi
        tracker.clear()
        object_counter.clear()
        object_speed.clear()
        history.clear()
        process_video(video_path, roi)
    except AttributeError:
        messagebox.showerror("Error", "No video selected yet!")
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error:\n{e}")

# Tkinter Window
root = tk.Tk()
root.title("YOLOv8 ROI-Based Object Detection")
root.geometry('400x300')
root.configure(bg='#2c3e50')

# Set font styles
header_font = tkfont.Font(family='Helvetica', size=18, weight='bold')
button_font = tkfont.Font(family='Helvetica', size=12)

# Header
header_label = tk.Label(root, text="YOLOv8 ROI Object Detection", font=header_font, fg='white', bg='#2c3e50')
header_label.pack(pady=20)

# Button Frame
button_frame = tk.Frame(root, bg='#2c3e50')
button_frame.pack(pady=30)

select_button = tk.Button(button_frame, text="Select Video", font=button_font, width=20, bg='#3498db', fg='white', command=select_video)
select_button.grid(row=0, column=0, pady=10)

start_button = tk.Button(button_frame, text="Start Processing", font=button_font, width=20, bg='#27ae60', fg='white', state=tk.DISABLED, command=start_processing)
start_button.grid(row=1, column=0, pady=10)

root.mainloop()
