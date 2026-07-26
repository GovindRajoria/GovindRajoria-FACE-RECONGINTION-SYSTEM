"""Order-of-magnitude CPU cost of LBPH training and prediction.

Runs against synthetic greyscale crops, so it says nothing at all about
accuracy — only about how much CPU the algorithm needs, which is what the
LBPH-over-embeddings decision in docs/DESIGN.md turns on and which is
independent of what the pixels contain.

The interesting column is prediction: LBPH compares a probe against every
stored histogram, so per-face cost grows with the size of the gallery.

    python scripts/lbph_cost.py
"""

from __future__ import annotations

import platform
import time

import cv2
import numpy as np

CROP_SIZE = 200
SAMPLES_PER_PERSON = 120
PREDICT_REPEATS = 200


def synthetic_crops(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, (CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        for _ in range(count)
    ]


def main() -> None:
    print(
        f"{'people':>7}  {'crops':>6}  {'train':>9}  {'predict':>12}"
    )
    for people in (1, 3, 10):
        faces: list[np.ndarray] = []
        labels: list[int] = []
        for person in range(people):
            faces.extend(synthetic_crops(SAMPLES_PER_PERSON, person))
            labels.extend([person] * SAMPLES_PER_PERSON)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        started = time.perf_counter()
        recognizer.train(faces, np.array(labels))
        train_seconds = time.perf_counter() - started

        probe = synthetic_crops(1, 999)[0]
        started = time.perf_counter()
        for _ in range(PREDICT_REPEATS):
            recognizer.predict(probe)
        predict_ms = (time.perf_counter() - started) / PREDICT_REPEATS * 1000

        print(
            f"{people:>7}  {len(faces):>6}  {train_seconds:>7.2f} s  "
            f"{predict_ms:>9.2f} ms"
        )

    print(
        f"\n{CROP_SIZE}x{CROP_SIZE} crops, {SAMPLES_PER_PERSON} per person, "
        f"prediction averaged over {PREDICT_REPEATS} calls."
    )
    print(f"OpenCV {cv2.__version__}, Python {platform.python_version()}")
    print(f"{platform.system()} {platform.release()} on {platform.machine()}")


if __name__ == "__main__":
    main()
