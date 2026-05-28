
# GestureFlow – AI-Based Hand Gesture Automation System

GestureFlow is a real-time computer vision–based system that enables users to control their computer using **hand gestures**.
By leveraging **MediaPipe**, **OpenCV**, and **Python automation libraries**, the system replaces traditional input devices like a mouse and keyboard with natural hand movements.

---

## 🚀 Features

- 🖱️ Cursor Control using index finger tracking
- 👆 Left Click using index + middle finger gesture
- 🤏 Zoom In / Zoom Out using pinch gestures
- 👉 Swipe Left / Right for navigation
- 🔊 Volume Control using open/closed palm
- 🪟 Window Management (Minimize / Maximize)
- 🎵 Media Controls (Play/Pause, Next Track)
- 🖼️ Screenshot Capture using multi-finger gesture
- ⚡ High FPS real-time performance
- 🧠 Debounce logic to avoid accidental triggers

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe Hands
- NumPy
- pynput
- PyAutoGUI
- Threading & Queues

---

## 🧩 Gesture Mapping

| Gesture | Action |
|------|------|
| Index Finger Up | Cursor Movement |
| Index + Middle | Left Click |
| Thumb + Index (Pinch In) | Zoom Out |
| Thumb + Index (Pinch Out) | Zoom In |
| Middle Finger Swipe | Swipe Left / Right |
| All Fingers Open | Volume Up |
| All Fingers Closed | Volume Down |
| Pinky Finger Up | App Switch |
| Index + Pinky | Maximize Window |
| Middle + Ring | Minimize Window |
| Thumb + Middle | Play / Pause |
| Thumb + Ring | Next Track |
| Index + Middle + Ring | Screenshot |

---

## ⚙️ Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/Naveen-YN/GestureFlow-AI-Based-Hand-Gesture-Automation-System.git
cd GestureFlow-AI-Based-Hand-Gesture-Automation-System
```

### Install Dependencies
```bash
pip install opencv-python mediapipe numpy pynput pyautogui screeninfo
```

### Run the Application
```bash
python main.py
```

---

## 🧠 How It Works

1. Webcam captures real-time video frames
2. MediaPipe detects hand landmarks
3. Gesture patterns are identified
4. System actions are triggered using automation libraries
5. Cursor movement is smoothed for stability

---
