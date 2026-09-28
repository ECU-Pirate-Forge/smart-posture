"""Normalized, confidence-aware adapter between YOLO Pose and analysis.

The internal coordinate system is normalized image coordinates: ``(0, 0)``
is the upper-left and ``(1, 1)`` is the lower-right.  Pixels are accepted at
the boundary when an ``image_size`` is supplied; calculations never depend on
the source resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from body_parts import BodyPart, LeftHip, LeftShoulder, RightHip, RightShoulder


YOLOV8_POSE_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


class Viewpoint(str, Enum):
    """Coarse, intentionally conservative image-plane viewpoint estimate."""

    FRONT_OR_BACK = "front_or_back"
    OBLIQUE = "oblique"
    PROFILE = "profile"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ViewpointEstimate:
    viewpoint: Viewpoint
    confidence: float
    reason: str


RawKeypoints = Union[Sequence[Sequence[float]], Mapping[str, Any], Any]


class Skeleton:
    """A full COCO-17 skeleton made of :class:`BodyPart` objects.

    ``keypoints`` may be an ``(17, 2)`` or ``(17, 3)`` array-like where the
    third value is confidence, a mapping of names to ``(x, y[, confidence])``,
    or an Ultralytics ``Keypoints`` object.  For Ultralytics objects, ``xyn``
    is preferred so no image dimensions are needed.  Missing or low-confidence
    points remain accessible but are marked unreliable rather than discarded.
    """

    def __init__(
        self,
        keypoints: Optional[RawKeypoints] = None,
        *,
        image_size: Optional[tuple[int, int]] = None,
        minimum_confidence: float = 0.5,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        self.minimum_confidence = minimum_confidence
        self.body_parts: dict[str, BodyPart] = {
            name: self._make_part(name) for name in YOLOV8_POSE_NAMES
        }
        if keypoints is not None:
            self.update(keypoints, image_size=image_size)

    @staticmethod
    def _make_part(name: str) -> BodyPart:
        specific = {
            "left_shoulder": LeftShoulder,
            "right_shoulder": RightShoulder,
            "left_hip": LeftHip,
            "right_hip": RightHip,
        }.get(name)
        return specific() if specific else BodyPart(label=name)

    def __getitem__(self, name: str) -> BodyPart:
        return self.body_parts[name]

    def get(self, name: str) -> Optional[BodyPart]:
        return self.body_parts.get(name)

    def __getattr__(self, name: str) -> BodyPart:
        # Attribute access keeps analyzer call sites readable, e.g.
        # skeleton.left_shoulder. Unknown attributes preserve normal errors.
        parts = self.__dict__.get("body_parts", {})
        if name in parts:
            return parts[name]
        raise AttributeError(name)

    def landmarks(self) -> Mapping[str, BodyPart]:
        """Read-only-by-convention mapping of standard YOLO landmark names."""
        return self.body_parts

    def reliable(self, *names: str) -> bool:
        """True only when all named landmarks are present and reliable."""
        return all(
            name in self.body_parts
            and self.body_parts[name].is_reliable(self.minimum_confidence)
            for name in names
        )

    def as_landmark_dict(self, include_unreliable: bool = False) -> dict[str, tuple[float, float]]:
        """Return normalized coordinates for renderers or other adapters."""
        result: dict[str, tuple[float, float]] = {}
        for name, part in self.body_parts.items():
            if part.position is None or (not include_unreliable and not part.is_reliable(self.minimum_confidence)):
                continue
            result[name] = (float(part.position[0]), float(part.position[1]))
        return result

    def update(self, keypoints: RawKeypoints, *, image_size: Optional[tuple[int, int]] = None) -> None:
        """Replace landmarks from a supported YOLO-style representation."""
        for name, values in self._iter_keypoints(keypoints):
            if name not in self.body_parts:
                continue
            position, confidence = self._parse_values(values, image_size)
            part = self.body_parts[name]
            part.position = position
            part.confidence = confidence

    @classmethod
    def from_ultralytics(cls, keypoints: Any, *, minimum_confidence: float = 0.5) -> "Skeleton":
        """Create a skeleton from ``result.keypoints`` or an equivalent object."""
        return cls(keypoints, minimum_confidence=minimum_confidence)

    def _iter_keypoints(self, keypoints: RawKeypoints) -> Iterable[tuple[str, Any]]:
        if isinstance(keypoints, Mapping):
            return keypoints.items()
        # Ultralytics Keypoints.xyn is normalized and usually shaped (1,17,2).
        values = getattr(keypoints, "xyn", keypoints)
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError("Skeleton accepts one detected person at a time")
            array = array[0]
        if array.ndim != 2 or array.shape[0] > len(YOLOV8_POSE_NAMES):
            raise ValueError("keypoints must have shape (N, 2) or (N, 3)")
        # ``Keypoints.xyn`` contains x/y only; detector confidence lives in
        # ``Keypoints.conf``. Preserve it instead of silently treating every
        # normalized point as perfect-confidence.
        confidence_values = getattr(keypoints, "conf", None)
        if confidence_values is not None and array.shape[1] == 2:
            if hasattr(confidence_values, "cpu"):
                confidence_values = confidence_values.cpu()
            if hasattr(confidence_values, "numpy"):
                confidence_values = confidence_values.numpy()
            confidence_array = np.asarray(confidence_values, dtype=float)
            if confidence_array.ndim == 2:
                if confidence_array.shape[0] != 1:
                    raise ValueError("Skeleton accepts one detected person at a time")
                confidence_array = confidence_array[0]
            if confidence_array.shape == (array.shape[0],):
                array = np.column_stack((array, confidence_array))
        return zip(YOLOV8_POSE_NAMES, array)

    @staticmethod
    def _parse_values(values: Any, image_size: Optional[tuple[int, int]]) -> tuple[Optional[Sequence[float]], float]:
        if values is None:
            return None, 0.0
        if isinstance(values, BodyPart):
            return values.position, values.confidence
        raw = np.asarray(values, dtype=float).reshape(-1)
        if len(raw) < 2 or not np.isfinite(raw[:2]).all():
            return None, 0.0
        x, y = float(raw[0]), float(raw[1])
        confidence = float(raw[2]) if len(raw) >= 3 and np.isfinite(raw[2]) else 1.0
        if image_size is not None:
            width, height = image_size
            if width <= 0 or height <= 0:
                raise ValueError("image_size must be (positive width, positive height)")
            x, y = x / width, y / height
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= confidence <= 1.0):
            return None, 0.0
        return (x, y), confidence

    def estimate_viewpoint(self) -> ViewpointEstimate:
        """Estimate view class from apparent shoulder/hip widths only.

        This is not a 3D reconstruction. It deliberately reports low
        confidence when the available 2D geometry cannot support a conclusion.
        """
        names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not self.reliable(*names):
            return ViewpointEstimate(Viewpoint.UNKNOWN, 0.0, "reliable shoulders and hips are required")
        shoulder_width = BodyPart.distance(self.left_shoulder.position, self.right_shoulder.position)  # type: ignore[arg-type]
        hip_width = BodyPart.distance(self.left_hip.position, self.right_hip.position)  # type: ignore[arg-type]
        torso_height = BodyPart.distance(
            BodyPart.midpoint(self.left_shoulder.position, self.right_shoulder.position),  # type: ignore[arg-type]
            BodyPart.midpoint(self.left_hip.position, self.right_hip.position),  # type: ignore[arg-type]
        )
        width_ratio = min(shoulder_width, hip_width) / max(torso_height, 1e-9)
        if width_ratio < 0.18:
            return ViewpointEstimate(Viewpoint.PROFILE, 0.65, "both apparent torso widths are narrow")
        if width_ratio < 0.35:
            return ViewpointEstimate(Viewpoint.OBLIQUE, 0.55, "apparent torso width suggests an oblique view")
        return ViewpointEstimate(Viewpoint.FRONT_OR_BACK, 0.6, "apparent torso width supports a frontal/back view")
