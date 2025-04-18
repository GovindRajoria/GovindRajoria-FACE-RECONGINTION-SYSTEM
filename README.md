Face Recognition System
A Python-based AI system for real-time face capture, training, and recognition using OpenCV.

Key Features
✔ Three-Step Workflow:

Face Capture (face_taker.py) – Collects 120 training images per user.

Model Training (face_trainer.py) – Uses OpenCV’s LBPH algorithm.

Real-Time Recognition (face_recognizer.py) – Identifies faces with confidence scores.

✔ User-Friendly CLI: Guided prompts for seamless interaction.
✔ Scalable: Add unlimited users by re-running the capture script.
✔ Lightweight: No GPU required (Haar Cascade + LBPH).


Quick Start
Capture Faces:

bash
python src/face_taker.py  
Enter your name when prompted.

Ensure good lighting and varied angles.

Train the Model:

bash
python src/face_trainer.py  
Run Recognition:

bash
python src/face_recognizer.py  
Press ESC to exit.

Dependencies
Python 3.x

OpenCV (pip install opencv-python)

NumPy (pip install numpy)

Pillow (pip install pillow)

Future Enhancements
Integrate deep learning (e.g., FaceNet for higher accuracy).

Deploy as a web app using Flask/Streamlit.

Add support for encrypted user data.
