# Design notes

Why this is built the way it is. The [README](../README.md) covers what the
code does and how to run it; this covers the decisions behind it, including the
ones that look wrong until you know the constraint.

## The constraint

The target is a machine with no GPU, no ML framework installed, and no
willingness to send face images anywhere. That single constraint decides almost
everything below.

## LBPH rather than face embeddings

The modern answer is an embedding model — FaceNet, ArcFace, a MobileFaceNet
variant — producing a 128- or 512-dimensional vector per face, compared by
cosine distance. It is more accurate, and materially more robust to pose and
lighting. It was not chosen, for three reasons.

**Enrolment cost.** LBPH "training" is histogram extraction; it needs no
optimiser, no epochs, and no GPU. Measured with `scripts/lbph_cost.py`:

| People | Crops | Train | Predict |
|---|---|---|---|
| 1 | 120 | 0.35 s | 4.2 ms/face |
| 3 | 360 | 1.05 s | 6.7 ms/face |
| 10 | 1,200 | 3.49 s | 15.9 ms/face |

*200×200 synthetic greyscale crops, 120 per person, OpenCV 5.0.0, Python 3.10 on
an Intel x86-64 laptop. Prediction averaged over 200 calls. Synthetic pixels, so
this measures CPU cost only — it says nothing about accuracy. Reproduce with
`python scripts/lbph_cost.py`.*

Adding a person is a sub-second retrain of the whole model. With embeddings the
enrolment story is better still — one forward pass, no retrain — but that is the
one place embeddings win on cost, and it comes attached to the next two points.

**Dependency weight.** LBPH ships inside `opencv-contrib-python`, which this
project already needs for capture and detection. An embedding model means
PyTorch or TensorFlow or an ONNX runtime, plus the weights, plus a
version-matched toolchain. That is a several-hundred-megabyte install and a
class of "works on my machine" failure this project does not otherwise have.

**Nothing has to leave the machine.** No model download, no hub token, no
service. `pip install -r requirements.txt` is the whole supply chain.

### What that costs, stated plainly

LBPH is a texture descriptor. It is sensitive to illumination and pose in a way
embeddings are not, and it has no notion of a face beyond the local binary
patterns in the crop it was given. Recognition quality between "trained under
office lighting" and "used in a dim room" degrades noticeably. No parameter
change fixes that; capturing samples across varied conditions helps most.

### The scaling limit, which is the real reason to switch

Read the prediction column again: **4.2 ms at one person, 15.9 ms at ten.** LBPH
compares a probe histogram against every stored histogram, so per-face cost is
linear in gallery size. At a hundred people it is no longer a real-time
algorithm on one core.

Embeddings do not have this shape. A single forward pass produces one vector,
and lookup against a hundred or a million enrolled faces is an index query. The
crossover is not about accuracy — it is that LBPH's cost model is wrong for a
gallery, and right for the handful of people this project targets.

**So: if this ever needs to recognise more than a few dozen people, the answer
is not to tune LBPH. It is to replace it.**

## Haar cascades rather than a DNN face detector

Same reasoning, one step earlier in the pipeline. OpenCV ships a DNN face
detector (`cv2.dnn`, an SSD ResNet-10) that outperforms Haar on profiles and
partial occlusion. It also needs a weights file and a prototxt fetched from
somewhere.

The Haar cascade is a 940 KB XML file **committed to this repository**, so
detection works on a fresh clone with no network. Given LBPH downstream — which
wants a frontal, roughly aligned crop anyway — a frontal-only detector is not
the binding constraint on accuracy. The recogniser is.

`scale_factor` and `min_neighbors` in `src/settings/settings.py` are the two
knobs that matter, and they trade recall against false positives directly.

## Three scripts rather than one application

Capture, train and recognise are separate entry points with no shared process
and no shared state beyond files on disk. This is not laziness about structure —
it is that the three have genuinely different lifetimes. Capture runs once per
person. Training runs after capture. Recognition runs continuously. Fusing them
into one program with modes would add a layer whose only job is to route between
three things that never run at the same time.

The cost is that `PATHS` in the settings module is a set of relative paths
resolved against the working directory, so the scripts must be run from the
repository root. `scripts/verify_setup.py` resolves paths against the repository
root instead, which is why it works from anywhere.

## The confidence number is inverted

`recognizer.predict()` returns a **distance**, where lower is a better match.
Reporting that number directly as "confidence" would show 30% for a good match
and 90% for a bad one. `src/face_recognizer.py` inverts it before display.

Predictions past the threshold are reported as *Unknown* rather than snapped to
the nearest label. A recogniser with no reject option assigns every face it sees
to somebody, which is the failure mode that makes this kind of system unsafe.

## Nothing ships pre-trained

`images/`, `trainer.yml` and `names.json` are git-ignored, and the development
sample set was removed from the repository history. The reasoning is in the
README under [A note on the data](../README.md#a-note-on-the-data) and is not
repeated here — the short version is that a face is not a credential you can
rotate after exposure.

The design consequence worth noting here: because nothing pre-trained ships,
there is no way to demo this repository without first enrolling a real person.
That is deliberate friction, not an oversight.

## What CI verifies

`ruff` with an explicitly pinned rule set, `compileall` over both source trees,
and `scripts/verify_setup.py` — which asserts `cv2.face` exists (the entire
reason `requirements.txt` names the contrib build) and that the committed
cascade loads as a usable classifier rather than the empty one
`CascadeClassifier` returns without raising.

**What it does not verify:** anything involving a camera, a trained model, or
recognition accuracy. A runner has no webcam and this repository ships no
enrolled faces, so no automated test exercises the pipeline end to end. That gap
is structural, not an omission — closing it would mean committing biometric data,
which is the one thing this project will not do.
