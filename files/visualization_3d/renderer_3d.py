"""
renderer_3d.py

3D counterpart to visualization/renderer.py, built on PyVista instead of
OpenCV. Same responsibility boundary as the 2D renderer: this module
ONLY builds and displays/exports 3D geometry from landmark data. It does
not perform pose estimation, posture analysis, angle math, file loading
of landmark data, or persistence beyond writing out the rendered result.

Key difference from the 2D renderer: because PyVista scenes have a real
camera, `render()` returns a `pv.Plotter` rather than a flat image array.
The caller then chooses what to do with it -- `.show()` for an
interactive orbit/pan/zoom window, or `save()` / `export_mesh()` below
for a static screenshot or a mesh file that can be opened in a full DCC
tool like Blender.

Depth handling: landmark values may be given as either (x, y) or
(x, y, z), normalized to [0, 1]. When z is omitted, every landmark is
placed on the same flat plane (`constants_3d.DEFAULT_FLAT_DEPTH`) -- the
figure has no real depth data, but still exists in true 3D space, so
bones render as properly oriented cylinders and the scene is fully
orbitable.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import pyvista as pv

if TYPE_CHECKING:
    from posture_analysis import PostureMetrics

from visualization.connections import Connection, SKELETON_CONNECTIONS

from . import constants_3d as constants
from .colors_3d import (
    GREEN,
    ORANGE,
    YELLOW,
    ColorRGB,
    ColorScheme3D,
    DEFAULT_COLOR_SCHEME_3D,
)

Landmark2D = Tuple[float, float]
Landmark3D = Tuple[float, float, float]
Landmark = Union[Landmark2D, Landmark3D]
LandmarkDict = Dict[str, Landmark]
WorldPoint = Tuple[float, float, float]


@dataclass
class RenderConfig3D:
    """
    Bundles all visual/style/scaling settings for the 3D renderer.

    Attributes:
        scale: Multiplier applied to normalized coordinates to produce
            world-space coordinates.
        flat_depth: z value used for landmarks supplied as (x, y) only.
        joint_radius: Sphere radius for joints, in world units.
        bone_radius: Cylinder ("tube") radius for bones, in world units.
        joint_resolution / bone_resolution: Mesh smoothness for spheres
            and tubes respectively. Higher = smoother, more triangles.
        color_scheme: A ColorScheme3D controlling background/joint/bone
            colors (with per-joint/per-bone override support).
        background_color: Scene background color (name or RGB tuple).
        window_size: (width, height) of the render window in pixels.
        connections: The list of (start, end) landmark-name pairs to
            draw as bones. Defaults to the full body skeleton.
    """

    scale: float = constants.DEFAULT_SCALE
    flat_depth: float = constants.DEFAULT_FLAT_DEPTH
    joint_radius: float = constants.DEFAULT_JOINT_RADIUS
    bone_radius: float = constants.DEFAULT_BONE_RADIUS
    joint_resolution: int = constants.DEFAULT_JOINT_RESOLUTION
    bone_resolution: int = constants.DEFAULT_BONE_RESOLUTION
    color_scheme: ColorScheme3D = field(default_factory=lambda: DEFAULT_COLOR_SCHEME_3D)
    background_color: str = constants.DEFAULT_BACKGROUND_COLOR
    window_size: Tuple[int, int] = constants.DEFAULT_WINDOW_SIZE
    connections: List[Connection] = field(
        default_factory=lambda: list(SKELETON_CONNECTIONS)
    )
    show_posture_guides: bool = True
    midpoint_radius: float = 0.16
    spine_radius: float = 0.06
    midpoint_color: ColorRGB = GREEN
    spine_midpoint_color: ColorRGB = YELLOW
    spine_color: ColorRGB = ORANGE


class Renderer3D:
    """
    Builds a navigable 3D stick-figure skeleton from landmark data.

    Typical usage:
        renderer = Renderer3D()
        renderer.show(landmarks)                     # interactive window
        renderer.save(landmarks, "skeleton.png")      # static screenshot
        renderer.export_mesh(landmarks, "skeleton.obj")  # open in Blender

    Like the 2D renderer, this class is stateless with respect to
    landmark data -- all per-call state lives in local variables inside
    each method. Configuration lives on the instance via
    `RenderConfig3D`, so one renderer can be reused across many
    skeletons/frames with consistent styling.
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

        Args:
            landmarks: Mapping of landmark name -> (x, y) or (x, y, z)
                normalized coordinates. Missing/invalid entries are
                skipped, same as the 2D renderer.
            connections: Optional override of which bones to draw.
            off_screen: Pass True for headless rendering (e.g. when only
                calling `.screenshot()` on a machine with no display).

        Returns:
            A `pv.Plotter` with the skeleton meshes already added. The
            caller decides what to do with it: `.show()` to open an
            interactive orbit/pan/zoom window, or `.screenshot(path)`
            for a static image.
        """
        plotter = pv.Plotter(
            off_screen=off_screen,
            window_size=list(self.config.window_size),
        )
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

        Exists as a separate step (rather than folding into `render()`)
        so multiple skeletons can be composed into a single scene later
        -- e.g. a side-by-side posture comparison -- by calling this
        once per skeleton on a shared Plotter, each with its own
        `Renderer3D` instance/`ColorScheme3D` for distinct coloring.
        """
        active_connections = (
            connections if connections is not None else self.config.connections
        )
        world_points = self._project_landmarks(landmarks)

        self._add_bones(plotter, world_points, active_connections)
        self._add_joints(plotter, world_points)
        self._add_posture_guides(plotter, posture_metrics)

    def show(
        self,
        landmarks: LandmarkDict,
        connections: Optional[List[Connection]] = None,
        posture_metrics: Optional["PostureMetrics"] = None,
    ) -> None:
        """Open an interactive window: left-drag to orbit, scroll to zoom, right-drag to pan."""
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
        Merge the skeleton's joints and bones into a single mesh and
        write it to disk (e.g. 'skeleton.obj', 'skeleton.ply').

        This is the escape hatch for "actually open it in Blender"
        rather than using this renderer's own interactive viewer --
        the exported file can be imported directly into Blender or any
        other 3D tool. Color information is not preserved by all
        formats (.obj in particular); geometry always is.
        """
        active_connections = (
            connections if connections is not None else self.config.connections
        )
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
        (y=0 is the top of frame, e.g. the head) while a 3D scene with
        an upright figure reads more naturally with y increasing
        upward. z passes through if supplied, otherwise every landmark
        is placed on `config.flat_depth`.
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
        """
        Return True if `coord` is a well-formed (x, y) or (x, y, z)
        tuple with all components in [0, 1] (epsilon-padded).
        """
        if coord is None:
            return False
        if len(coord) not in (2, 3):
            return False

        lo = constants.COORDINATE_MIN - constants.COORDINATE_EPSILON
        hi = constants.COORDINATE_MAX + constants.COORDINATE_EPSILON
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
        return (
            point[0] * self.config.scale,
            (1.0 - point[1]) * self.config.scale,
            self.config.flat_depth * self.config.scale,
        )

    def _add_posture_guides(
        self, plotter: pv.Plotter, posture_metrics: Optional["PostureMetrics"]
    ) -> None:
        """Draw analyzer-produced torso midpoints and their spine segment."""
        if not self.config.show_posture_guides or posture_metrics is None:
            return
        shoulder = posture_metrics.shoulder_midpoint
        hip = posture_metrics.hip_midpoint
        spine_midpoint = posture_metrics.spine_midpoint
        if shoulder is not None:
            plotter.add_mesh(
                pv.Sphere(radius=self.config.midpoint_radius, center=self._normalised_to_world((float(shoulder[0]), float(shoulder[1])))),
                color=self.config.midpoint_color,
            )
        if hip is not None:
            plotter.add_mesh(
                pv.Sphere(radius=self.config.midpoint_radius, center=self._normalised_to_world((float(hip[0]), float(hip[1])))),
                color=self.config.midpoint_color,
            )
        if spine_midpoint is not None:
            plotter.add_mesh(
                pv.Sphere(radius=self.config.midpoint_radius, center=self._normalised_to_world((float(spine_midpoint[0]), float(spine_midpoint[1])))),
                color=self.config.spine_midpoint_color,
            )
        if shoulder is not None and hip is not None:
            spine = pv.Line(
                self._normalised_to_world((float(hip[0]), float(hip[1]))),
                self._normalised_to_world((float(shoulder[0]), float(shoulder[1]))),
            ).tube(radius=self.config.spine_radius, n_sides=self.config.bone_resolution)
            plotter.add_mesh(spine, color=self.config.spine_color, smooth_shading=True)

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

            color: ColorRGB = self.config.color_scheme.color_for_bone(connection)
            tube = self._make_bone_mesh(start, end)
            plotter.add_mesh(tube, color=color, smooth_shading=True)

    def _add_joints(
        self,
        plotter: pv.Plotter,
        world_points: Dict[str, WorldPoint],
    ) -> None:
        """Add all joint sphere meshes to the plotter."""
        for name, point in world_points.items():
            color: ColorRGB = self.config.color_scheme.color_for_joint(name)
            sphere = self._make_joint_mesh(point)
            plotter.add_mesh(sphere, color=color, smooth_shading=True)
