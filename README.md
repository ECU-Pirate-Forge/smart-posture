# SmartPosture

SmartPosture is an internal MVP for assessing basic body alignment from a
single image or video frame using YOLOv8 Pose keypoints. It is designed to
make the reliability of image-plane posture measurements explicit rather than
presenting uncertain 2D measurements as precise 3D facts.

## How it works

```text
Input image/video → YOLOv8 Pose → normalized keypoints → Skeleton
                                                        → BodyPart objects
                                                        → PostureAnalyzer
                                                        → metrics + reliability
```

`Skeleton` is the integration boundary: it maps the standard COCO 17 YOLOv8
Pose keypoints to consistently named `BodyPart` instances. `PostureAnalyzer`
then consumes shoulders and hips to calculate shoulder/hip tilt, spine
inclination, torso area, and torso symmetry. The included 3D-rendering
prototypes are visualization-only; they do not infer depth or calculate
posture.

## Coordinate system and depth limitations

The application uses normalized image coordinates internally (`x` and `y` in
`[0, 1]`, origin at the upper-left). Pass pixel keypoints with `image_size`
and they are normalized at the boundary, so calculations are independent of
source image resolution. Convert to pixels only in a visualization layer.

Standard YOLOv8 Pose exposes 2D keypoints (and confidence), not true metric
depth. `BodyPart.depth` is intentionally optional for a future 3D provider;
the current MVP never fabricates it. A conservative apparent-width viewpoint
estimate labels the body as front/back, oblique, profile, or unknown. In
oblique/profile views, affected 2D torso metrics stay available but are marked
`estimated` with reduced reliability in `metrics.metric_results`.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv venv
```

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

Install Ultralytics separately when running inference; it is deliberately not
required for core geometry tests because `Skeleton` accepts its keypoint output
without importing the model package.

## Usage

```python
from skeleton import Skeleton
from posture_analysis import PostureAnalyzer

# Equivalent to result.keypoints.xyn[0] for one YOLO detection.
keypoints = {
    "left_shoulder": (0.40, 0.30, 0.95),
    "right_shoulder": (0.60, 0.30, 0.94),
    "left_hip": (0.43, 0.70, 0.95),
    "right_hip": (0.57, 0.70, 0.94),
}
skeleton = Skeleton(keypoints, minimum_confidence=0.5)
metrics = PostureAnalyzer.analyze(skeleton)

print(metrics.spine_inclination)
print(metrics.metric_results["spine_inclination"])
print(metrics.viewpoint, metrics.viewpoint_confidence)
```

The 3D renderer can draw the analyzer's calculated shoulder/hip midpoints
(green), central spine midpoint (yellow), and spine segment (orange) when
passed the same metrics object:

```python
renderer.save(skeleton.as_landmark_dict(), "posture.png", posture_metrics=metrics)
```

This overlay consumes the analyzer result; it does not duplicate midpoint or
spine calculations inside the renderer.

For pixel coordinates, use `Skeleton(pixel_keypoints, image_size=(width,
height))`. `Skeleton.from_ultralytics(result.keypoints)` accepts an Ultralytics
Keypoints object for a single person and prefers its normalized `xyn` data.

### Run YOLO Pose on a source

`run_pose.py` is the reusable command-line entry point for an image, video,
URL, or webcam. It selects the highest-confidence detected person in each
frame and emits one JSON record per usable frame.

```bash
python run_pose.py input.mp4 --output posture_metrics.jsonl
python run_pose.py input.jpg --render-first-3d posture_overlay.png
python run_pose.py 0 --output webcam_metrics.jsonl
```

Use `--model path/to/best.pt` for a custom pose checkpoint. The optional 3D
overlay requires the existing PyVista renderer dependency.

## Project structure

```text
body_parts.py       Landmark representation and geometry helpers
skeleton.py         YOLO/COCO keypoint parsing, normalization, viewpoint flag
posture_analysis.py Four-landmark posture metrics and reliability metadata
3dRender/           Standalone PyVista visualization prototype
files/              Visualization prototype materials
tests/              Core unit tests
```

## Status and next steps

Implemented: normalized keypoint ingestion, complete COCO-17 landmark access,
low-confidence/missing-keypoint handling, four existing upper-body metrics,
and viewpoint-aware reliability flags. This is not a clinical tool and cannot
measure true forward-head posture, body rotation, or depth from one 2D frame.

Practical next steps are temporal smoothing and person tracking for video,
calibrated multi-camera or depth-model input, richer head/neck metrics when a
reliable profile view is present, a maintained 2D visualizer, and validation
against labelled posture data.
