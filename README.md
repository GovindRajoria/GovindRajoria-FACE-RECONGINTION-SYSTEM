
---

# 🧠 Face Recognition System

A Python-based AI system for **real-time face capture**, **model training**, and **face recognition** using OpenCV. Designed to be lightweight, scalable, and easy to use — no GPU required!

---

## 🔑 Key Features

✅ **Three-Step Workflow**  
- **📸 Face Capture** (`face_taker.py`) — Collects 120 training images per user.  
- **🧠 Model Training** (`face_trainer.py`) — Trains using OpenCV’s LBPH algorithm.  
- **🎯 Real-Time Recognition** (`face_recognizer.py`) — Identifies faces and shows confidence scores.

✅ **User-Friendly CLI** — Guided prompts for seamless interaction.  
✅ **Scalable** — Add unlimited users by re-running the capture script.  
✅ **Lightweight** — Works on CPU using Haar Cascades + LBPH (no GPU needed).

---

## 🚀 Quick Start

### 1. Capture Faces
```bash
python src/face_taker.py
```
> 📝 Enter your name when prompted.  
> 💡 Ensure good lighting and varied angles for better accuracy.

### 2. Train the Model
```bash
python src/face_trainer.py
```

### 3. Run Real-Time Recognition
```bash
python src/face_recognizer.py
```
> ⎋ Press **ESC** to exit.

---

## 📦 Dependencies

- Python 3.x  
- [OpenCV](https://pypi.org/project/opencv-contrib-python/)  
  ```bash
  pip install opencv-contrib-python
  ```
- NumPy  
  ```bash
  pip install numpy
  ```
- Pillow  
  ```bash
  pip install pillow
  ```

---

## 🔮 Future Enhancements

- 🤖 Integrate deep learning (e.g., **FaceNet** for higher accuracy)  
- 🌐 Deploy as a web app using **Flask** or **Streamlit**  
- 🔒 Add support for **encrypted user data**  

---
