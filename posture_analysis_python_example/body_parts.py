from typing import Optional, Sequence, Tuple
import numpy as np
import math
Vec2 = np.ndarray


# ===========================================================================
# BodyPart – base class
# ===========================================================================

class BodyPart:
    """
    Abstract base class for a single anatomical landmark.

    Parameters
    ----------
    position : array-like of shape (2,) or None
        (x, y) coordinates in image space (float, 0-1 for normalised YOLO
        output, or pixels).  Pass ``None`` to mark the keypoint as missing.
    confidence : float, optional
        Keypoint detection confidence in [0, 1].
    label : str, optional
        Human-readable name used in ``__repr__``.
    """

    def __init__(
        self,
        position: Optional[Sequence[float]] = None,
        confidence: float = 1.0,
        label: str = "BodyPart",
    ) -> None:
        self._position: Optional[Vec2] = (
            np.array(position, dtype=np.float64) if position is not None else None
        )
        self._confidence = float(confidence)
        self._label = label

        # Derived attributes (computed lazily or set externally)
        self._posture: Optional[str] = None   # e.g. "elevated", "neutral", …
        self._angle: Optional[float] = None   # degrees, relative to some reference

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> Optional[Vec2]:
        """(x, y) position as a NumPy array, or None if the keypoint is missing."""
        return self._position

    @position.setter
    def position(self, value: Optional[Sequence[float]]) -> None:
        self._position = (
            np.array(value, dtype=np.float64) if value is not None else None
        )

    @property
    def confidence(self) -> float:
        """Detection confidence in [0, 1]."""
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        self._confidence = float(value)

    @property
    def posture(self) -> Optional[str]:
        """
        Categorical posture descriptor (e.g. 'elevated', 'neutral', 'dropped').
        Populated externally by PostureAnalyzer or subclass logic.
        """
        return self._posture

    @posture.setter
    def posture(self, value: Optional[str]) -> None:
        self._posture = value

    @property
    def angle(self) -> Optional[float]:
        """
        Angle (degrees) associated with this landmark relative to a reference
        direction.  Populated externally by PostureAnalyzer or subclass logic.
        """
        return self._angle

    @angle.setter
    def angle(self, value: Optional[float]) -> None:
        self._angle = value

    @property
    def is_valid(self) -> bool:
        """True when the keypoint position is available (not None)."""
        return self._position is not None
    
    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pos_str = (
            f"({self._position[0]:.4f}, {self._position[1]:.4f})"
            if self._position is not None
            else "None"
        )
        return (
            f"{self.__class__.__name__}("
            f"position={pos_str}, "
            f"confidence={self._confidence:.2f}, "
            f"posture={self._posture!r}, "
            f"angle={self._angle})"
        )
    

    @staticmethod
    def distance(a: Vec2, b: Vec2) -> float:
        """
        Euclidean distance between two 2-D points.

        Parameters
        ----------
        a, b : np.ndarray, shape (2,)

        Returns
        -------
        float
        """
        return float(np.linalg.norm(b - a))

    @staticmethod
    def midpoint(a: Vec2, b: Vec2) -> Vec2:
        """
        Midpoint between two 2-D points.

        Parameters
        ----------
        a, b : np.ndarray, shape (2,)

        Returns
        -------
        np.ndarray, shape (2,)
        """
        return (a + b) * 0.5

    @staticmethod
    def angle_between_vectors(u: Vec2, v: Vec2) -> float:
        """
        Angle (degrees) between two 2-D vectors using the dot-product formula.
        Returns 0 if either vector has zero magnitude.

        Parameters
        ----------
        u, v : np.ndarray, shape (2,)

        Returns
        -------
        float  – angle in [0, 180]
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        cos_theta = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
        return float(math.degrees(math.acos(cos_theta)))

    @staticmethod
    def line_angle(a: Vec2, b: Vec2) -> float:
        """
        Signed angle (degrees) that the line segment a→b makes with the
        positive x-axis, in the range (-180, 180].

        Parameters
        ----------
        a, b : np.ndarray, shape (2,)  – start and end points

        Returns
        -------
        float
        """
        delta = b - a
        return float(math.degrees(math.atan2(delta[1], delta[0])))

    @staticmethod
    def triangle_area(p1: Vec2, p2: Vec2, p3: Vec2) -> float:
        """
        Signed area of the triangle formed by three 2-D points, computed via
        the cross-product formula (positive when vertices are counter-clockwise).

        Parameters
        ----------
        p1, p2, p3 : np.ndarray, shape (2,)

        Returns
        -------
        float  – signed area (use abs() for the unsigned value)
        """
        # |( p2 - p1 ) × ( p3 - p1 )| / 2
        v1 = p2 - p1
        v2 = p3 - p1
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        return cross * 0.5

   

class LeftShoulder(BodyPart):
    """Left shoulder keypoint (YOLO index 5)."""

    def __init__(
        self,
        position: Optional[Sequence[float]] = None,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(position=position, confidence=confidence, label="LeftShoulder")


class RightShoulder(BodyPart):
    """Right shoulder keypoint (YOLO index 6)."""

    def __init__(
        self,
        position: Optional[Sequence[float]] = None,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(position=position, confidence=confidence, label="RightShoulder")


class LeftHip(BodyPart):
    """Left hip keypoint (YOLO index 11)."""

    def __init__(
        self,
        position: Optional[Sequence[float]] = None,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(position=position, confidence=confidence, label="LeftHip")


class RightHip(BodyPart):
    """Right hip keypoint (YOLO index 12)."""

    def __init__(
        self,
        position: Optional[Sequence[float]] = None,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(position=position, confidence=confidence, label="RightHip")


