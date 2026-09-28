"""Run SmartPosture on an image, video, URL, or webcam with YOLO Pose.

Examples
--------
python run_pose.py input.mp4 --output posture_metrics.jsonl
python run_pose.py input.jpg --render-first-3d posture_overlay.png
python run_pose.py 0 --output webcam_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from posture_analysis import PostureAnalyzer
from skeleton import Skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one person per YOLO Pose frame.")
    parser.add_argument("source", help="Image/video path, URL, directory, or webcam index (for example: 0).")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics pose checkpoint path or model name.")
    parser.add_argument("--detection-confidence", type=float, default=0.25, help="Minimum person-detection confidence.")
    parser.add_argument("--keypoint-confidence", type=float, default=0.5, help="Minimum keypoint confidence for posture metrics.")
    parser.add_argument("--output", type=Path, help="Optional JSON Lines output path; one record per processed frame.")
    parser.add_argument("--render-first-3d", type=Path, metavar="PNG", help="Save a 3D posture overlay for the first usable pose only.")
    return parser.parse_args()


def choose_source(source: str) -> str | int:
    """Treat a numeric source as a webcam index, otherwise pass it through."""
    return int(source) if source.isdigit() else source


def select_person(result: Any) -> int | None:
    """Select the highest-confidence person in one Ultralytics result frame."""
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    return int(result.boxes.conf.argmax().item())


def metric_record(frame_index: int, result: Any, person_index: int, metrics: Any) -> dict[str, Any]:
    """Convert NumPy-backed posture output to a JSON-serializable record."""
    return {
        "frame_index": frame_index,
        "source": str(getattr(result, "path", "")),
        "person_index": person_index,
        "viewpoint": metrics.viewpoint,
        "viewpoint_confidence": metrics.viewpoint_confidence,
        "shoulder_midpoint": metrics.shoulder_midpoint.tolist() if metrics.shoulder_midpoint is not None else None,
        "hip_midpoint": metrics.hip_midpoint.tolist() if metrics.hip_midpoint is not None else None,
        "spine_midpoint": metrics.spine_midpoint.tolist() if metrics.spine_midpoint is not None else None,
        "metrics": {
            name: {
                "value": result.value,
                "confidence": result.confidence,
                "valid": result.valid,
                "reason": result.reason,
                "estimated": result.estimated,
            }
            for name, result in metrics.metric_results.items()
        },
    }


def save_first_3d_overlay(path: Path, skeleton: Skeleton, metrics: Any) -> None:
    """Render one representative frame without making PyVista mandatory otherwise."""
    renderer_dir = Path(__file__).resolve().parent / "3dRender"
    sys.path.insert(0, str(renderer_dir))
    from skeleton_3d import SkeletonRenderer3D

    path.parent.mkdir(parents=True, exist_ok=True)
    SkeletonRenderer3D().save(
        skeleton.as_landmark_dict(),
        str(path),
        posture_metrics=metrics,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.detection_confidence <= 1.0 or not 0.0 <= args.keypoint_confidence <= 1.0:
        raise SystemExit("Confidence values must be between 0 and 1.")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install inference support first: python -m pip install ultralytics") from error

    output = args.output.open("w", encoding="utf-8") if args.output else None
    overlay_saved = False
    try:
        model = YOLO(args.model)
        results = model(
            choose_source(args.source),
            stream=True,
            conf=args.detection_confidence,
        )
        for frame_index, result in enumerate(results):
            person_index = select_person(result)
            if person_index is None:
                continue

            skeleton = Skeleton.from_ultralytics(
                result.keypoints[person_index],
                minimum_confidence=args.keypoint_confidence,
            )
            metrics = PostureAnalyzer.analyze(skeleton)
            record = metric_record(frame_index, result, person_index, metrics)
            line = json.dumps(record)
            print(line)
            if output:
                output.write(line + "\n")

            if args.render_first_3d and not overlay_saved:
                save_first_3d_overlay(args.render_first_3d, skeleton, metrics)
                overlay_saved = True
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
