# Face Recognition System

Real-time face recognition on CPU using OpenCV — Haar cascade detection for
locating faces, LBPH (Local Binary Patterns Histograms) for identifying them.
No GPU, no deep learning framework, no cloud service.

The point of the design is cost: LBPH trains in seconds on a hundred or so
grayscale crops and recognises in a single pass over the frame, which makes it
viable on hardware where a CNN-based embedding model would not be.

---

## How it works

Three stages, each a standalone script.

### 1. Capture — `src/face_taker.py`

Prompts for a name, allocates the next free numeric ID by scanning existing
filenames, and records the mapping into `names.json`. Then it opens the webcam
and, on every frame, runs the Haar cascade over the grayscale image and writes
each detected face region to `images/Users-{id}-{n}.jpg`, overlaying capture
progress on the preview until 120 samples are collected.

Only the cropped grayscale face region is stored, never the full frame.

### 2. Train — `src/face_trainer.py`

Loads every image in `images/`, derives the label from the filename, and fits
`cv2.face.LBPHFaceRecognizer` over the set. The trained model is written to
`trainer.yml`.

### 3. Recognise — `src/face_recognizer.py`

Loads `trainer.yml` and `names.json`, then for each detected face predicts a
label and a distance. LBPH returns *distance*, where lower means a better match,
so the displayed confidence is inverted from it; predictions past the threshold
are reported as unknown rather than forced to the nearest label.

---

## Setup

Requires Python 3.8+ and a webcam.

```bash
git clone https://github.com/GovindRajoria/face-recognition-system.git
cd face-recognition-system
pip install -r requirements.txt
```

`opencv-contrib-python` is required rather than `opencv-python` — the `cv2.face`
module that provides LBPH ships only in the contrib build.

Run the stages in order, from the repository root:

```bash
python src/face_taker.py       # capture 120 samples, prompts for a name
python src/face_trainer.py     # fit the LBPH model -> trainer.yml
python src/face_recognizer.py  # live recognition; ESC to quit
```

Add more people by re-running the capture step and retraining.

## Configuration

Everything tunable lives in `src/settings/settings.py`: camera index and
resolution, Haar cascade `scale_factor` / `min_neighbors` / `min_size`, the
number of training samples per person, and the data paths.

If detection is missing faces, `scale_factor` (default `1.3`) and
`min_neighbors` (default `5`) are the two knobs that matter — lower values
detect more aggressively at the cost of false positives.

---

## A note on the data

`images/`, `trainer.yml` and `names.json` are git-ignored, and the sample set
this project was developed against has been removed from the repository history.

Face images and a model trained on them are biometric data. They identify a
specific person permanently — unlike a password, they cannot be rotated after
exposure. They belong on the machine that captured them and nowhere else, which
is why nothing here ships with a pre-trained model: run the capture step and
generate your own.

If you extend this to other people, get their consent first.

---

## Limitations

- LBPH is sensitive to lighting and pose. Accuracy degrades noticeably between
  the conditions it was trained under and materially different ones — capturing
  samples across varied lighting and angles helps more than any parameter change.
- Haar cascades detect frontal faces only; profiles are missed.
- There is no anti-spoofing. A photograph held up to the camera will be
  recognised as the person in it, so this is unsuitable for access control on
  its own.
- Recognition runs single-threaded, inline with frame capture.

## Possible extensions

- Swap LBPH for a face-embedding model (FaceNet, ArcFace) and weigh the accuracy
  gain against the CPU cost.
- Add liveness detection — blink or texture analysis — before trusting a match.
- Persist recognition events with timestamps to build an attendance log.

## Author

**Govind Kumar** — AI/ML Developer, Metro Infrasys Private Limited
[GitHub](https://github.com/GovindRajoria) · govindrajoria97@gmail.com
