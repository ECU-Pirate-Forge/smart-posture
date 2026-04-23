"""
posture_analysis.py
===================
Posture analysis using YOLO-style pose keypoints.

Architecture
------------
BodyPart          – abstract base class (position, posture, angle)
  ├─ LeftShoulder
  ├─ RightShoulder
  ├─ LeftHip
  └─ RightHip
PostureAnalyzer   – aggregates four BodyPart instances and computes metrics

Coordinate conventions
----------------------
YOLO keypoints are given in *image-space* (x right, y down, origin top-left).
PostureAnalyzer optionally normalises coordinates to a *body-centred* system:
  - Origin   : hip midpoint
  - Scale    : distance from hip midpoint to shoulder midpoint = 1.0
  - Rotation : hip line is made horizontal (optional)
This makes all metrics independent of subject position, scale, and in-plane
rotation within the frame.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from body_parts import *
import body_parts


# ===========================================================================
# PostureMetrics – lightweight data container
# ===========================================================================

class PostureMetrics:
    """
    Container for all posture metrics returned by ``PostureAnalyzer``.

    All angle values are in **degrees**.  Any metric that cannot be computed
    due to missing keypoints is stored as ``None``.

    Attributes
    ----------
    shoulder_tilt : float or None
        Angle (°) of the shoulder line w.r.t. horizontal.
        0° = perfectly level; positive = left shoulder higher in image-space
        (remember: y increases downward for image coordinates).
    hip_tilt : float or None
        Angle (°) of the hip line w.r.t. horizontal.
    spine_inclination : float or None
        Angle (°) between the torso vector (hip midpoint → shoulder midpoint)
        and the vertical axis.  0° = upright.
    torso_symmetry : float or None
        Difference in area (body-centred units²) between the two triangles
        formed by splitting the torso quadrilateral along the spine.
        Values near 0 indicate good left-right symmetry.
    torso_area : float or None
        Total area of the torso quadrilateral (LS → RS → RH → LH) in
        body-centred units².
    shoulder_midpoint : np.ndarray or None
        Midpoint of the two shoulders in *image* space.
    hip_midpoint : np.ndarray or None
        Midpoint of the two hips in *image* space.
    """

    def __init__(self) -> None:
        self.shoulder_tilt:     Optional[float]    = None
        self.hip_tilt:          Optional[float]    = None
        self.spine_inclination: Optional[float]    = None
        self.torso_symmetry:    Optional[float]    = None
        self.torso_area:        Optional[float]    = None
        self.shoulder_midpoint: Optional[Vec2]     = None
        self.hip_midpoint:      Optional[Vec2]     = None

    def __repr__(self) -> str:
        lines = ["PostureMetrics("]
        for attr, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                val_str = f"({val[0]:.4f}, {val[1]:.4f})"
            elif isinstance(val, float):
                val_str = f"{val:.4f}°" if "midpoint" not in attr else f"{val:.4f}"
            else:
                val_str = repr(val)
            lines.append(f"  {attr} = {val_str}")
        lines.append(")")
        return "\n".join(lines)


# ===========================================================================
# PostureAnalyzer
# ===========================================================================

class PostureAnalyzer:
    """
    Analyses upper-body posture from four anatomical landmarks.

    Parameters
    ----------
    left_shoulder  : LeftShoulder
    right_shoulder : RightShoulder
    left_hip       : LeftHip
    right_hip      : RightHip
    align_hips     : bool, default True
        If True, normalised coordinates are also rotated so that the hip line
        is horizontal (hip tilt removed from the coordinate system). This
        makes ``spine_inclination`` purely about forward/backward lean.

    Notes
    -----
    Body-centred normalisation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Let:
        H  = hip midpoint        (origin of body-centred frame)
        S  = shoulder midpoint
        d  = ||S - H||           (body scale = 1 unit)

    Normalised point of landmark P:
        P_norm = (P - H) / d

    Optional rotation to make hips horizontal:
        θ     = angle of hip line w.r.t. positive x-axis
        R(−θ) rotates coordinates so the hip line aligns with x-axis.

    Image-space convention (y down)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    YOLO outputs (x, y) with y increasing downward.  Spine vector in image
    space points *upward*, i.e. has a *negative* y component when the person
    is upright.  We account for this when comparing to the vertical.
    """

    # Vertical reference in image space (pointing up, i.e. negative y)
    _VERTICAL_UP: Vec2 = np.array([0.0, -1.0])

    def __init__(
        self,
        left_shoulder:  body_parts.LeftShoulder,
        right_shoulder: body_parts.RightShoulder,
        left_hip:       body_parts.LeftHip,
        right_hip:      body_parts.RightHip,
        align_hips:     bool = True,
    ) -> None:
        self.left_shoulder  = left_shoulder
        self.right_shoulder = right_shoulder
        self.left_hip       = left_hip
        self.right_hip      = right_hip
        self.align_hips     = align_hips

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, *parts: BodyPart) -> bool:
        """Return True only if every supplied BodyPart has a valid position."""
        return all(p.is_valid for p in parts)

    def _normalised_coords(
        self,
    ) -> Optional[Tuple[Vec2, Vec2, Vec2, Vec2]]:
        """
        Return (ls, rs, lh, rh) in the body-centred coordinate system.
        Returns None if any of the four keypoints is missing or if the
        shoulder-to-hip distance is zero (degenerate pose).
        """
        if not self._require(
            self.left_shoulder, self.right_shoulder,
            self.left_hip, self.right_hip
        ):
            return None

        ls = self.left_shoulder.position
        rs = self.right_shoulder.position
        lh = self.left_hip.position
        rh = self.right_hip.position

        # Body-centred origin and scale
        hip_mid  = BodyPart.midpoint(lh, rh)        # type: ignore[arg-type]
        shld_mid = BodyPart.midpoint(ls, rs)         # type: ignore[arg-type]
        scale    = BodyPart.distance(hip_mid, shld_mid)

        if scale < 1e-9:
            return None   # degenerate (all points collapsed)

        # Translate and scale
        ls_n = (ls - hip_mid) / scale   # type: ignore[operator]
        rs_n = (rs - hip_mid) / scale
        lh_n = (lh - hip_mid) / scale
        rh_n = (rh - hip_mid) / scale

        # Optional rotation: make hip line horizontal
        if self.align_hips:
            # Angle of left_hip → right_hip vector
            hip_vec = rh_n - lh_n
            theta   = math.atan2(float(hip_vec[1]), float(hip_vec[0]))
            cos_t, sin_t = math.cos(-theta), math.sin(-theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            ls_n, rs_n, lh_n, rh_n = (R @ v for v in (ls_n, rs_n, lh_n, rh_n))

        return ls_n, rs_n, lh_n, rh_n   # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Individual metric methods
    # ------------------------------------------------------------------

    def shoulder_tilt(self) -> Optional[float]:
        """
        Angle (°) of the shoulder line relative to horizontal.

        Computed in **image space** (before body-centred normalisation) so
        that the raw tilt is preserved.  Positive = left shoulder is *lower*
        in the image (higher y value) than right shoulder.

        Returns
        -------
        float or None
        """
        if not self._require(self.left_shoulder, self.right_shoulder):
            return None
        ls = self.left_shoulder.position
        rs = self.right_shoulder.position
        return BodyPart.line_angle(ls, rs)   # type: ignore[arg-type]

    def hip_tilt(self) -> Optional[float]:
        """
        Angle (°) of the hip line relative to horizontal (image space).

        Returns
        -------
        float or None
        """
        if not self._require(self.left_hip, self.right_hip):
            return None
        lh = self.left_hip.position
        rh = self.right_hip.position
        return BodyPart.line_angle(lh, rh)   # type: ignore[arg-type]

    def spine_inclination(self) -> Optional[float]:
        """
        Angle (°) between the torso vector (hip midpoint → shoulder midpoint)
        and the upward vertical [0, -1] in image space.

        0° means perfectly upright; larger values indicate leaning.

        Returns
        -------
        float or None
        """
        if not self._require(
            self.left_shoulder, self.right_shoulder,
            self.left_hip, self.right_hip
        ):
            return None

        shld_mid = BodyPart.midpoint(
            self.left_shoulder.position,   # type: ignore[arg-type]
            self.right_shoulder.position,  # type: ignore[arg-type]
        )
        hip_mid = BodyPart.midpoint(
            self.left_hip.position,   # type: ignore[arg-type]
            self.right_hip.position,  # type: ignore[arg-type]
        )

        torso_vec = shld_mid - hip_mid   # points from hips toward shoulders

        if np.linalg.norm(torso_vec) < 1e-9:
            return None

        return BodyPart.angle_between_vectors(torso_vec, self._VERTICAL_UP)

    def torso_triangle_symmetry(self) -> Optional[float]:
        """
        Absolute difference in area between the two triangles formed by the
        torso quadrilateral split along its diagonal spine.

        Triangle 1 : LeftShoulder  – RightShoulder – LeftHip
        Triangle 2 : RightShoulder – RightHip       – LeftHip

        Areas are computed in the **body-centred** coordinate system so the
        result is scale-invariant.  A value near 0 indicates good symmetry.

        Returns
        -------
        float or None
        """
        norm = self._normalised_coords()
        if norm is None:
            return None

        ls, rs, lh, rh = norm

        area1 = abs(BodyPart.triangle_area(ls, rs, lh))
        area2 = abs(BodyPart.triangle_area(rs, rh, lh))

        return abs(area1 - area2)

    def torso_triangle_area(self) -> Optional[float]:
        """
        Total area of the torso quadrilateral (LS → RS → RH → LH) computed
        as the sum of its two diagonal triangles in the **body-centred** frame.

        Returns
        -------
        float or None
        """
        norm = self._normalised_coords()
        if norm is None:
            return None

        ls, rs, lh, rh = norm

        area1 = abs(BodyPart.triangle_area(ls, rs, lh))
        area2 = abs(BodyPart.triangle_area(rs, rh, lh))

        return area1 + area2
    

    # ------------------------------------------------------------------
    # Convenience: compute everything at once
    # ------------------------------------------------------------------

    def analyse(self) -> PostureMetrics:
        """
        Compute all available posture metrics and return them in a
        ``PostureMetrics`` instance.  Also sets the ``angle`` and ``posture``
        attributes on each BodyPart landmark.

        Returns
        -------
        PostureMetrics
        """
        metrics = PostureMetrics()

        # Raw tilt metrics (image space)
        metrics.shoulder_tilt     = self.shoulder_tilt()
        metrics.hip_tilt          = self.hip_tilt()
        metrics.spine_inclination = self.spine_inclination()

        # Scale-normalised metrics (body-centred space)
        metrics.torso_symmetry    = self.torso_triangle_symmetry()
        metrics.torso_area        = self.torso_triangle_area()

        # Convenience midpoints in image space
        if self._require(self.left_shoulder, self.right_shoulder):
            metrics.shoulder_midpoint = BodyPart.midpoint(
                self.left_shoulder.position,   # type: ignore[arg-type]
                self.right_shoulder.position,  # type: ignore[arg-type]
            )
        if self._require(self.left_hip, self.right_hip):
            metrics.hip_midpoint = BodyPart.midpoint(
                self.left_hip.position,   # type: ignore[arg-type]
                self.right_hip.position,  # type: ignore[arg-type]
            )

        # ---------------------------------------------------------------
        # Annotate BodyPart instances with derived angle / posture labels
        # ---------------------------------------------------------------
        shld_tilt = metrics.shoulder_tilt
        if shld_tilt is not None:
            # For shoulders, we use the tilt relative to each midpoint.
            # Left shoulder: positive tilt angle means it's lower than right.
            self.left_shoulder.angle  = shld_tilt
            self.right_shoulder.angle = shld_tilt

            # Simple categorical labels (thresholds are heuristic)
            if abs(shld_tilt) < 2.0:
                label = "level"
            elif shld_tilt > 0:
                label = "left_low"
            else:
                label = "right_low"

            self.left_shoulder.posture  = label
            self.right_shoulder.posture = label

        hip_t = metrics.hip_tilt
        if hip_t is not None:
            self.left_hip.angle  = hip_t
            self.right_hip.angle = hip_t

            if abs(hip_t) < 2.0:
                label = "level"
            elif hip_t > 0:
                label = "left_low"
            else:
                label = "right_low"

            self.left_hip.posture  = label
            self.right_hip.posture = label

        return metrics


# ===========================================================================
# Example / demo
# ===========================================================================

def _run_example() -> None:
    """Demonstrate class instantiation and posture metric output."""
    print("=" * 60)
    print("  Posture Analysis – Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Instantiate landmarks (normalised YOLO coordinates, x/y in [0,1])
    # ------------------------------------------------------------------
    left_shoulder  = LeftShoulder( (0.45, 0.30))
    right_shoulder = RightShoulder((0.55, 0.32))
    left_hip       = LeftHip(      (0.46, 0.60))
    right_hip      = RightHip(     (0.54, 0.61))

    print("\n--- Landmarks (before analysis) ---")
    for part in (left_shoulder, right_shoulder, left_hip, right_hip):
        print(" ", part)

    # ------------------------------------------------------------------
    # 2. Run the analyser
    # ------------------------------------------------------------------
    analyser = PostureAnalyzer(
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        align_hips=True,
    )
    metrics = analyser.analyse()

    # ------------------------------------------------------------------
    # 3. Print results
    # ------------------------------------------------------------------
    print("\n--- Posture Metrics ---")
    print(metrics)

    print("\n--- Landmarks (after analysis, with angle & posture) ---")
    for part in (left_shoulder, right_shoulder, left_hip, right_hip):
        print(" ", part)

    # ------------------------------------------------------------------
    # 4. Static utility demos
    # ------------------------------------------------------------------
    print("\n--- Static Utility Method Demos ---")
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    print(f"  distance([0,0], [3,4])             = {BodyPart.distance(a, b):.4f}")
    print(f"  midpoint([0,0], [3,4])             = {BodyPart.midpoint(a, b)}")
    print(f"  line_angle([0,0], [1,0])           = {BodyPart.line_angle(a, np.array([1.0, 0.0])):.1f}°")
    print(f"  angle_between([1,0], [0,1])        = {BodyPart.angle_between_vectors(np.array([1.0,0.0]), np.array([0.0,1.0])):.1f}°")
    p1, p2, p3 = np.array([0.,0.]), np.array([1.,0.]), np.array([0.,1.])
    print(f"  triangle_area([0,0],[1,0],[0,1])   = {BodyPart.triangle_area(p1, p2, p3):.4f}")

    # ------------------------------------------------------------------
    # 5. Missing-keypoint robustness demo
    # ------------------------------------------------------------------
    print("\n--- Robustness: missing right_hip ---")
    analyser_partial = PostureAnalyzer(
        left_shoulder, right_shoulder, left_hip,
        RightHip(None),   # explicitly missing
    )
    m2 = analyser_partial.analyse()
    print(f"  shoulder_tilt     = {m2.shoulder_tilt}")
    print(f"  hip_tilt          = {m2.hip_tilt}      ← None (right hip missing)")
    print(f"  spine_inclination = {m2.spine_inclination}      ← None (right hip missing)")
    print(f"  torso_symmetry    = {m2.torso_symmetry}      ← None (right hip missing)")
    print(f"  torso_area        = {m2.torso_area}      ← None (right hip missing)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _run_example()
