"""
skeleton_3d.py

Standalone 3D skeleton visualization for SmartPosture.

Self-contained on purpose: everything needed to render a navigable 3D
stick-figure skeleton -- topology, colors, configuration, and the
renderer itself -- lives in this one file. No import from any other
project module (only the third-party `pyvista` package is required).

Responsibility boundary (unchanged from the original design): this
module ONLY draws. It receives a dict of named landmarks and turns it
into 3D geometry. It does not do YOLO inference, posture analysis,
angle math, file loading of landmark data, or persistence beyond
writing out its own rendered result.

Install:
    pip install pyvista

Usage:
    from skeleton_3d import SkeletonRenderer3D

    landmarks = {
        "nose": (0.52, 0.11),
        "left_shoulder": (0.44, 0.31),
        "right_shoulder": (0.58, 0.31),
        # ...
    }

    renderer = SkeletonRenderer3D()
    renderer.show(landmarks)                          # interactive: orbit/pan/zoom
    renderer.save(landmarks, "skeleton.png")           # static screenshot
    renderer.export_mesh(landmarks, "skeleton.obj")    # open in real Blender
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import pyvista as pv

if TYPE_CHECKING:
    from posture_analysis import PostureMetrics

# ===========================================================================
# Skeleton topology
# ===========================================================================
# Pure data: which pairs of landmark names are joined by a "bone". Landmark
# names follow the standard YOLOv8-Pose (COCO 17-keypoint) convention.

Connection = Tuple[str, str]

FACE_CONNECTIONS: List[Connection] = [
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]

ARM_CONNECTIONS: List[Connection] = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

TORSO_CONNECTIONS: List[Connection] = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

LEG_CONNECTIONS: List[Connection] = [
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

SKELETON_CONNECTIONS: List[Connection] = [
    *FACE_CONNECTIONS,
    *ARM_CONNECTIONS,
    *TORSO_CONNECTIONS,
    *LEG_CONNECTIONS,
]

# ===========================================================================
# Colors
# ===========================================================================
# RGB, 0-1 float each -- PyVista's convention (not OpenCV's 0-255 BGR).

ColorRGB = Tuple[float, float, float]

WHITE: ColorRGB = (1.0, 1.0, 1.0)
BLACK: ColorRGB = (0.0, 0.0, 0.0)
RED: ColorRGB = (1.0, 0.0, 0.0)
GREEN: ColorRGB = (0.0, 0.78, 0.0)
BLUE: ColorRGB = (0.0, 0.0, 1.0)
YELLOW: ColorRGB = (0.86, 0.86, 0.0)
GRAY: ColorRGB = (0.5, 0.5, 0.5)
ORANGE: ColorRGB = (1.0, 0.55, 0.0)


@dataclass
class ColorScheme3D:
    """
    Colors for a single skeleton render call.

    `joint_overrides` / `bone_overrides` aren't used by default, but are
    checked on every draw call -- future "colored joints" / "colored
    bones" features are just populating these dicts, no renderer
    changes required.
    """

    background: ColorRGB = WHITE
    joint: ColorRGB = RED
    bone: ColorRGB = BLUE
    joint_overrides: Dict[str, ColorRGB] = field(default_factory=dict)
    bone_overrides: Dict[Connection, ColorRGB] = field(default_factory=dict)

    def color_for_joint(self, joint_name: str) -> ColorRGB:
        return self.joint_overrides.get(joint_name, self.joint)

    def color_for_bone(self, connection: Connection) -> ColorRGB:
        return self.bone_overrides.get(connection, self.bone)


# ===========================================================================
# Constants / defaults
# ===========================================================================

DEFAULT_SCALE = 10.0          # normalized coords x this = world units
DEFAULT_FLAT_DEPTH = 0.0      # z used when a landmark has no z given

DEFAULT_JOINT_RADIUS = 0.22   # sphere radius, world units
DEFAULT_BONE_RADIUS = 0.09    # cylinder ("tube") radius, world units

DEFAULT_JOINT_RESOLUTION = 16
DEFAULT_BONE_RESOLUTION = 16

DEFAULT_BACKGROUND_COLOR = "white"
DEFAULT_WINDOW_SIZE = (900, 900)

COORDINATE_MIN = 0.0
COORDINATE_MAX = 1.0
COORDINATE_EPSILON = 1e-6

# ===========================================================================
# Config
# ===========================================================================

Landmark2D = Tuple[float, float]
Landmark3D = Tuple[float, float, float]
Landmark = Union[Landmark2D, Landmark3D]
LandmarkDict = Dict[str, Landmark]
WorldPoint = Tuple[float, float, float]


@dataclass
class RenderConfig3D:
    """Bundles all visual/style/scaling settings for the renderer."""

    scale: float = DEFAULT_SCALE
    flat_depth: float = DEFAULT_FLAT_DEPTH
    joint_radius: float = DEFAULT_JOINT_RADIUS
    bone_radius: float = DEFAULT_BONE_RADIUS
    joint_resolution: int = DEFAULT_JOINT_RESOLUTION
    bone_resolution: int = DEFAULT_BONE_RESOLUTION
    color_scheme: ColorScheme3D = field(default_factory=ColorScheme3D)
    background_color: str = DEFAULT_BACKGROUND_COLOR
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE
    connections: List[Connection] = field(default_factory=lambda: list(SKELETON_CONNECTIONS))
    show_posture_guides: bool = True
    midpoint_radius: float = 0.16
    spine_radius: float = 0.06
    midpoint_color: ColorRGB = GREEN
    spine_midpoint_color: ColorRGB = YELLOW
    spine_color: ColorRGB = ORANGE


# ===========================================================================
# Renderer
# ===========================================================================


class SkeletonRenderer3D:
    """
    Builds a navigable 3D stick-figure skeleton from landmark data.

    Stateless with respect to landmark data -- all per-call state lives
    in local variables inside each method. Configuration lives on the
    instance via `RenderConfig3D`, so one renderer can be reused across
    many skeletons/frames with consistent styling.
    """

    def __init__(self, config: Optional[RenderConfig3D] = None) -> None:
        self.config = config or RenderConfig3D()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        landmarks: LandmarkDict,
        connections: Optional[List[Connection]] = None,
        off_screen: bool = False,
        posture_metrics: Optional["PostureMetrics"] = None,
    ) -> pv.Plotter:
        """
        Build a fresh Plotter containing the rendered skeleton.

        Returns the Plotter itself so the caller decides what to do with
        it: `.show()` for an interactive orbit/pan/zoom window, or
        `.screenshot(path)` for a static image.
        """
        plotter = pv.Plotter(off_screen=off_screen, window_size=list(self.config.window_size))
        plotter.set_background(self.config.background_color)
        self.add_to_plotter(plotter, landmarks, connections, posture_metrics)
        plotter.view_isometric()
        plotter.reset_camera()
        return plotter

    def add_to_plotter(
        self,
        plotter: pv.Plotter,
        landmarks: LandmarkDict,
        connections: Optional[List[Connection]] = None,
        posture_metrics: Optional["PostureMetrics"] = None,
    ) -> None:
        """
        Add one skeleton's meshes onto an existing Plotter.

        Kept separate from `render()` so multiple skeletons can be
        composed into a single scene later (e.g. a side-by-side posture
        comparison) by calling this once per skeleton on a shared
        Plotter, each with its own renderer instance/color scheme.
        """
        active_connections = connections if connections is not None else self.config.connections
        world_points = self._project_landmarks(landmarks)

        self._add_bones(plotter, world_points, active_connections)
        self._add_joints(plotter, world_points)
        self._add_posture_guides(plotter, posture_metrics)

    def show(self, landmarks: LandmarkDict, connections: Optional[List[Connection]] = None, posture_metrics: Optional["PostureMetrics"] = None) -> None:
        """Open an interactive window: left-drag orbits, scroll zooms, right-drag pans."""
        plotter = self.render(landmarks, connections, off_screen=False, posture_metrics=posture_metrics)
        plotter.show()

    def save(
        self,
        landmarks: LandmarkDict,
        path: str,
        connections: Optional[List[Connection]] = None,
        posture_metrics: Optional["PostureMetrics"] = None,
    ) -> None:
        """Render a single static screenshot to `path` (e.g. 'skeleton.png')."""
        plotter = self.render(landmarks, connections, off_screen=True, posture_metrics=posture_metrics)
        plotter.screenshot(path)
        plotter.close()

    def export_mesh(
        self,
        landmarks: LandmarkDict,
        path: str,
        connections: Optional[List[Connection]] = None,
    ) -> None:
        """
        Merge the skeleton's joints and bones into a single mesh and write
        it to disk (e.g. 'skeleton.obj', 'skeleton.ply') for import into
        an actual DCC tool like Blender. Color isn't preserved by all
        formats (.obj in particular); geometry always is.
        """
        active_connections = connections if connections is not None else self.config.connections
        world_points = self._project_landmarks(landmarks)

        meshes = []
        for connection in active_connections:
            start = world_points.get(connection[0])
            end = world_points.get(connection[1])
            if start is None or end is None:
                continue
            meshes.append(self._make_bone_mesh(start, end))

        for point in world_points.values():
            meshes.append(self._make_joint_mesh(point))

        if not meshes:
            raise ValueError("No valid landmarks/connections to export.")

        combined = pv.merge(meshes)
        combined.save(path)

    # ------------------------------------------------------------------
    # Internal helpers -- coordinate projection
    # ------------------------------------------------------------------

    def _project_landmarks(self, landmarks: LandmarkDict) -> Dict[str, WorldPoint]:
        """
        Convert normalized landmark coordinates into world-space points,
        dropping any that are missing or invalid.

        x maps directly to world x. y is inverted (world_y = 1 - y)
        because image-style normalized coordinates increase downward
        (y=0 is the top of frame / the head), while an upright figure in
        a 3D scene reads more naturally with y increasing upward. z
        passes through if supplied, otherwise every landmark sits on
        `config.flat_depth`.
        """
        world_points: Dict[str, WorldPoint] = {}
        for name, coord in landmarks.items():
            if not self._is_valid_landmark(coord):
                continue
            x, y = coord[0], coord[1]
            z = coord[2] if len(coord) == 3 else self.config.flat_depth

            world_points[name] = (
                x * self.config.scale,
                (1.0 - y) * self.config.scale,
                z * self.config.scale,
            )
        return world_points

    @staticmethod
    def _is_valid_landmark(coord: Optional[Landmark]) -> bool:
        """True if `coord` is a well-formed (x, y) or (x, y, z) tuple in [0, 1]."""
        if coord is None:
            return False
        if len(coord) not in (2, 3):
            return False

        lo = COORDINATE_MIN - COORDINATE_EPSILON
        hi = COORDINATE_MAX + COORDINATE_EPSILON
        return all(lo <= value <= hi for value in coord)

    # ------------------------------------------------------------------
    # Internal helpers -- mesh construction
    # ------------------------------------------------------------------

    def _make_bone_mesh(self, start: WorldPoint, end: WorldPoint) -> pv.PolyData:
        """Build a cylindrical tube mesh connecting two world points."""
        line = pv.Line(start, end)
        return line.tube(radius=self.config.bone_radius, n_sides=self.config.bone_resolution)

    def _make_joint_mesh(self, center: WorldPoint) -> pv.PolyData:
        """Build a sphere mesh at a world point."""
        return pv.Sphere(
            radius=self.config.joint_radius,
            center=center,
            theta_resolution=self.config.joint_resolution,
            phi_resolution=self.config.joint_resolution,
        )

    def _normalised_to_world(self, point: Tuple[float, float]) -> WorldPoint:
        """Project a normalized image-space point using the renderer's scale."""
        return (point[0] * self.config.scale, (1.0 - point[1]) * self.config.scale, self.config.flat_depth * self.config.scale)

    def _add_posture_guides(self, plotter: pv.Plotter, posture_metrics: Optional["PostureMetrics"]) -> None:
        """Draw analyzer-produced torso midpoints and their spine segment.

        This renderer deliberately does not recalculate midpoints: callers pass
        the ``PostureMetrics`` object produced by ``PostureAnalyzer``.
        """
        if not self.config.show_posture_guides or posture_metrics is None:
            return
        shoulder = posture_metrics.shoulder_midpoint
        hip = posture_metrics.hip_midpoint
        spine_midpoint = posture_metrics.spine_midpoint
        if shoulder is not None:
            plotter.add_mesh(pv.Sphere(radius=self.config.midpoint_radius, center=self._normalised_to_world((float(shoulder[0]), float(shoulder[1])))), color=self.config.midpoint_color)
        if hip is not None:
            plotter.add_mesh(pv.Sphere(radius=self.config.midpoint_radius, center=self._normalised_to_world((float(hip[0]), float(hip[1])))), color=self.config.midpoint_color)
        if spine_midpoint is not None:
            plotter.add_mesh(pv.Sphere(radius=self.config.midpoint_radius, center=self._normalised_to_world((float(spine_midpoint[0]), float(spine_midpoint[1])))), color=self.config.spine_midpoint_color)
        if shoulder is not None and hip is not None:
            plotter.add_mesh(pv.Line(self._normalised_to_world((float(hip[0]), float(hip[1]))), self._normalised_to_world((float(shoulder[0]), float(shoulder[1])))).tube(radius=self.config.spine_radius, n_sides=self.config.bone_resolution), color=self.config.spine_color, smooth_shading=True)

    def _add_bones(
        self,
        plotter: pv.Plotter,
        world_points: Dict[str, WorldPoint],
        active_connections: List[Connection],
    ) -> None:
        """Add all bone tube meshes to the plotter. Skips connections with a missing endpoint."""
        for connection in active_connections:
            start = world_points.get(connection[0])
            end = world_points.get(connection[1])
            if start is None or end is None:
                continue

            color = self.config.color_scheme.color_for_bone(connection)
            tube = self._make_bone_mesh(start, end)
            plotter.add_mesh(tube, color=color, smooth_shading=True)

    def _add_joints(self, plotter: pv.Plotter, world_points: Dict[str, WorldPoint]) -> None:
        """Add all joint sphere meshes to the plotter."""
        for name, point in world_points.items():
            color = self.config.color_scheme.color_for_joint(name)
            sphere = self._make_joint_mesh(point)
            plotter.add_mesh(sphere, color=color, smooth_shading=True)


# ===========================================================================
# Example usage (only runs when this file is executed directly)
# ===========================================================================

if __name__ == "__main__":
    # Keep this standalone visualizer runnable directly from its own folder
    # while using the project's shared skeleton/analyzer pipeline.
    from pathlib import Path
    import sys

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from posture_analysis import PostureAnalyzer
    from skeleton import Skeleton

    landmarks = {
        "nose": (0.52, 0.11),
        "left_eye": (0.49, 0.10),
        "right_eye": (0.55, 0.10),
        "left_shoulder": (0.44, 0.31),
        "right_shoulder": (0.58, 0.31),
        "left_elbow": (0.38, 0.45),
        "right_elbow": (0.64, 0.45),
        "left_wrist": (0.34, 0.58),
        "right_wrist": (0.68, 0.58),
        "left_hip": (0.46, 0.60),
        "right_hip": (0.56, 0.60),
        "left_knee": (0.45, 0.78),
        "right_knee": (0.57, 0.78),
        "left_ankle": (0.44, 0.95),
        "right_ankle": (0.58, 0.95),
    }

    skeleton = Skeleton(landmarks)
    posture_metrics = PostureAnalyzer.analyze(skeleton)

    renderer = SkeletonRenderer3D()
    # Green spheres are shoulder/hip midpoints, the yellow sphere is their
    # calculated spine midpoint, and the orange line is the torso reference.
    renderer.show(
        skeleton.as_landmark_dict(),
        posture_metrics=posture_metrics,
    )  # opens an interactive window
