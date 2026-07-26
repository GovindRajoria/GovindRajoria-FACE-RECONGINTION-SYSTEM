"""Check that the environment can actually run this project.

Three things go wrong far more often than anything else, and all three fail at
runtime with errors that do not name the real cause:

1. `opencv-python` installed instead of `opencv-contrib-python`. LBPH lives in
   `cv2.face`, which ships only in the contrib build, so the failure is an
   AttributeError several steps into training.
2. The Haar cascade XML missing or truncated. `CascadeClassifier` does not
   raise — it returns an empty classifier that silently detects nothing.
3. The settings module not importable, which usually means the script was run
   from the wrong directory.

Run it from anywhere:  python scripts/verify_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    sys.exit(1)


def main() -> None:
    try:
        from settings.settings import CAMERA, FACE_DETECTION, PATHS, TRAINING
    except ImportError as error:
        fail(f"could not import src/settings/settings.py: {error}")

    try:
        import cv2
    except ImportError as error:
        fail(f"OpenCV is not installed: {error}")

    print(f"OK    OpenCV {cv2.__version__}")

    if not hasattr(cv2, "face"):
        fail(
            "cv2.face is missing — this is the base opencv-python package. "
            "Install opencv-contrib-python (see requirements.txt)."
        )
    print("OK    cv2.face is present (contrib build)")

    try:
        cv2.face.LBPHFaceRecognizer_create()
    except Exception as error:  # noqa: BLE001 - any failure here is fatal
        fail(f"could not construct an LBPH recogniser: {error}")
    print("OK    LBPH recogniser constructs")

    cascade_path = REPO_ROOT / PATHS["cascade_file"]
    if not cascade_path.is_file():
        fail(f"cascade file not found at {cascade_path}")

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        fail(
            f"{cascade_path.name} loaded as an empty classifier — the file is "
            "present but not a usable cascade."
        )
    print(f"OK    {cascade_path.name} loads ({cascade_path.stat().st_size} bytes)")

    # Not a correctness check, just a readout: these are the values every stage
    # runs against, and getting them wrong is the usual cause of "it detects
    # nothing".
    print(
        f"      camera index {CAMERA['index']} at {CAMERA['width']}x{CAMERA['height']}, "
        f"scale_factor {FACE_DETECTION['scale_factor']}, "
        f"min_neighbors {FACE_DETECTION['min_neighbors']}, "
        f"{TRAINING['samples_needed']} samples per person"
    )
    print("\nEnvironment looks good. No camera was opened by this check.")


if __name__ == "__main__":
    main()
