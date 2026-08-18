import json
from pathlib import Path

import bpy
from mathutils import Vector


# ============================================================
# SECTION 1: SETTINGS
# ============================================================

# Replace this with the exact JSON path printed by your
# MediaPipe Python script.
JSON_PATH = Path(
    "/Users/nathanaligwekwe/Desktop/smart-posture/pose_avater_test/pose_world_landmarks.json"
)

# Increase this if the mannequin appears too small.
# MediaPipe world coordinates are relatively small because
# they represent real-world measurements.
AVATAR_SCALE = 2.0

# Size of the cylinders used for arms and legs.
LIMB_RADIUS = 0.045

# Size of the spheres used for joints.
JOINT_RADIUS = 0.06


# ============================================================
# SECTION 2: MEDIAPIPE LANDMARK NUMBERS
# ============================================================

# Face landmarks
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8

# Upper body landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Lower body landmarks
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ============================================================
# SECTION 3: BODY CONNECTIONS
# ============================================================

# Each tuple contains:
# (name of body part, starting landmark, ending landmark)
#
# Blender will create one cylinder between each pair.

BODY_SEGMENTS = [
    # Left arm
    ("UpperArm.L", LEFT_SHOULDER, LEFT_ELBOW),
    ("Forearm.L", LEFT_ELBOW, LEFT_WRIST),

    # Right arm
    ("UpperArm.R", RIGHT_SHOULDER, RIGHT_ELBOW),
    ("Forearm.R", RIGHT_ELBOW, RIGHT_WRIST),

    # Left leg
    ("Thigh.L", LEFT_HIP, LEFT_KNEE),
    ("Shin.L", LEFT_KNEE, LEFT_ANKLE),

    # Right leg
    ("Thigh.R", RIGHT_HIP, RIGHT_KNEE),
    ("Shin.R", RIGHT_KNEE, RIGHT_ANKLE),

    # Left foot
    ("Heel.L", LEFT_ANKLE, LEFT_HEEL),
    ("Foot.L", LEFT_HEEL, LEFT_FOOT_INDEX),

    # Right foot
    ("Heel.R", RIGHT_ANKLE, RIGHT_HEEL),
    ("Foot.R", RIGHT_HEEL, RIGHT_FOOT_INDEX),
]


# These landmarks will receive visible joint spheres.
JOINT_LANDMARKS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]


# ============================================================
# SECTION 4: CLEAR THE BLENDER SCENE
# ============================================================

def clear_scene():
    """
    Delete Blender's default cube, camera, light,
    and anything left from a previous run.
    """

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    print("Cleared the Blender scene.")


# ============================================================
# SECTION 5: LOAD THE MEDIAPIPE JSON FILE
# ============================================================

def load_landmarks(json_path):
    """
    Read the landmark JSON created by MediaPipe.
    """

    if not json_path.is_file():
        raise FileNotFoundError(
            f"Landmark JSON file was not found:\n{json_path}"
        )

    with json_path.open("r", encoding="utf-8") as json_file:
        pose_data = json.load(json_file)

    landmarks = pose_data.get("landmarks")

    if not landmarks:
        raise ValueError(
            "The JSON file does not contain any landmarks."
        )

    print(f"Loaded {len(landmarks)} landmarks.")

    return landmarks


# ============================================================
# SECTION 6: CONVERT MEDIAPIPE COORDINATES TO BLENDER
# ============================================================

def mediapipe_to_blender(landmark):
    """
    Convert one MediaPipe coordinate into Blender's
    coordinate system.

    MediaPipe:
        X = left/right
        Y = up/down
        Z = depth

    Blender:
        X = left/right
        Y = forward/backward
        Z = up/down

    The signs are changed so the person appears upright.
    """

    return Vector((
        float(landmark["x"]) * AVATAR_SCALE,
        -float(landmark["z"]) * AVATAR_SCALE,
        -float(landmark["y"]) * AVATAR_SCALE,
    ))


def build_point_dictionary(landmarks):
    """
    Convert the JSON landmark list into a dictionary.

    This lets us retrieve a landmark using its number:

        points[11] = left shoulder position
        points[13] = left elbow position
    """

    points = {}

    for landmark in landmarks:
        landmark_id = int(landmark["id"])

        points[landmark_id] = mediapipe_to_blender(
            landmark
        )

    return points


def move_avatar_to_ground(points):
    """
    MediaPipe coordinates are centered around the hips.

    This function moves the entire pose vertically so that
    its lowest landmark rests at Blender Z = 0.
    """

    lowest_z = min(point.z for point in points.values())

    vertical_offset = -lowest_z

    for landmark_id in points:
        points[landmark_id].z += vertical_offset

    print("Moved the avatar onto the ground plane.")


# ============================================================
# SECTION 7: CREATE MATERIALS
# ============================================================

def create_material(name, color, roughness=0.6):
    """
    Create a basic Blender material.

    The color is provided as:
        red, green, blue, alpha
    """

    material = bpy.data.materials.new(name=name)

    material.use_nodes = True
    material.diffuse_color = color

    # Find Blender's main shader node.
    principled_shader = material.node_tree.nodes.get(
        "Principled BSDF"
    )

    if principled_shader is not None:
        principled_shader.inputs[
            "Base Color"
        ].default_value = color

        principled_shader.inputs[
            "Roughness"
        ].default_value = roughness

    return material


def assign_material(blender_object, material):
    """
    Attach a material to a Blender mesh object.
    """

    blender_object.data.materials.append(material)


# ============================================================
# SECTION 8: CREATE JOINT SPHERES
# ============================================================

def create_joint(
    name,
    position,
    radius,
    material
):
    """
    Create one sphere at a landmark position.
    """

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=16,
        radius=radius,
        location=position,
    )

    joint = bpy.context.object
    joint.name = name

    assign_material(joint, material)

    # Make the sphere look smoother.
    for polygon in joint.data.polygons:
        polygon.use_smooth = True

    return joint


# ============================================================
# SECTION 9: CREATE A CYLINDER BETWEEN TWO LANDMARKS
# ============================================================

def create_limb(
    name,
    start_position,
    end_position,
    radius,
    material
):
    """
    Create a cylinder connecting two landmark positions.

    For example:
        shoulder -> elbow
        elbow -> wrist
        hip -> knee
    """

    # Vector pointing from the first joint to the second.
    direction = end_position - start_position

    # Length of the body segment.
    length = direction.length

    if length < 0.0001:
        print(f"Skipped {name}: segment was too short.")
        return None

    # The cylinder should be centered between the two joints.
    midpoint = (
        start_position + end_position
    ) / 2.0

    # Blender cylinders point along their local Z axis.
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=radius,
        depth=length,
        location=midpoint,
    )

    limb = bpy.context.object
    limb.name = name

    # Rotate the cylinder so its Z axis follows the
    # direction between the two landmarks.
    limb.rotation_mode = "QUATERNION"

    limb.rotation_quaternion = direction.to_track_quat(
        "Z",
        "Y",
    )

    assign_material(limb, material)

    for polygon in limb.data.polygons:
        polygon.use_smooth = True

    return limb


# ============================================================
# SECTION 10: CREATE AN ELLIPSOID BETWEEN TWO POSITIONS
# ============================================================

def create_ellipsoid_between(
    name,
    start_position,
    end_position,
    width,
    depth,
    material
):
    """
    Create an elongated sphere between two points.

    This is used for the torso because a torso should be
    wider than an arm or leg.
    """

    direction = end_position - start_position
    length = direction.length

    if length < 0.0001:
        return None

    midpoint = (
        start_position + end_position
    ) / 2.0

    # Start with a regular sphere.
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32,
        ring_count=20,
        radius=1.0,
        location=midpoint,
    )

    torso = bpy.context.object
    torso.name = name

    # Scale the sphere into an ellipsoid.
    torso.scale = (
        width,
        depth,
        length / 2.0,
    )

    # Rotate the ellipsoid to follow the spine direction.
    torso.rotation_mode = "QUATERNION"

    torso.rotation_quaternion = direction.to_track_quat(
        "Z",
        "Y",
    )

    assign_material(torso, material)

    for polygon in torso.data.polygons:
        polygon.use_smooth = True

    return torso


# ============================================================
# SECTION 11: BUILD THE MANNEQUIN
# ============================================================

def create_pose_avatar(points):
    """
    Construct the complete static mannequin from the
    MediaPipe landmark positions.
    """

    # Create a light material for the mannequin body.
    body_material = create_material(
        name="BodyMaterial",
        color=(0.62, 0.58, 0.48, 1.0),
        roughness=0.7,
    )

    # Create a dark material for the visible joints.
    joint_material = create_material(
        name="JointMaterial",
        color=(0.03, 0.03, 0.03, 1.0),
        roughness=0.5,
    )

    # --------------------------------------------------------
    # Create the arms, legs, and feet.
    # --------------------------------------------------------

    for name, start_id, end_id in BODY_SEGMENTS:
        create_limb(
            name=name,
            start_position=points[start_id],
            end_position=points[end_id],
            radius=LIMB_RADIUS,
            material=body_material,
        )

    # --------------------------------------------------------
    # Create visible spheres at the major joints.
    # --------------------------------------------------------

    for landmark_id in JOINT_LANDMARKS:
        create_joint(
            name=f"Joint_{landmark_id}",
            position=points[landmark_id],
            radius=JOINT_RADIUS,
            material=joint_material,
        )

    # --------------------------------------------------------
    # Calculate the torso center line.
    # --------------------------------------------------------

    shoulder_midpoint = (
        points[LEFT_SHOULDER]
        + points[RIGHT_SHOULDER]
    ) / 2.0

    hip_midpoint = (
        points[LEFT_HIP]
        + points[RIGHT_HIP]
    ) / 2.0

    shoulder_width = (
        points[LEFT_SHOULDER]
        - points[RIGHT_SHOULDER]
    ).length

    hip_width = (
        points[LEFT_HIP]
        - points[RIGHT_HIP]
    ).length

    # --------------------------------------------------------
    # Create the torso.
    # --------------------------------------------------------

    create_ellipsoid_between(
        name="Torso",
        start_position=hip_midpoint,
        end_position=shoulder_midpoint,
        width=shoulder_width * 0.48,
        depth=shoulder_width * 0.25,
        material=body_material,
    )

    # --------------------------------------------------------
    # Create the pelvis.
    # --------------------------------------------------------

    pelvis = create_joint(
        name="Pelvis",
        position=hip_midpoint,
        radius=1.0,
        material=body_material,
    )

    # Stretch the pelvis sphere into a pelvis shape.
    pelvis.scale = (
        hip_width * 0.55,
        hip_width * 0.32,
        hip_width * 0.30,
    )

    # --------------------------------------------------------
    # Estimate the center of the head.
    # --------------------------------------------------------

    if LEFT_EAR in points and RIGHT_EAR in points:
        head_center = (
            points[LEFT_EAR]
            + points[RIGHT_EAR]
        ) / 2.0
    else:
        head_center = points[NOSE]

    head_radius = max(
        shoulder_width * 0.22,
        0.12,
    )

    # --------------------------------------------------------
    # Create the neck.
    # --------------------------------------------------------

    direction_to_head = (
        head_center - shoulder_midpoint
    )

    if direction_to_head.length > 0.001:
        neck_end = (
            head_center
            - direction_to_head.normalized()
            * head_radius
            * 0.7
        )

        create_limb(
            name="Neck",
            start_position=shoulder_midpoint,
            end_position=neck_end,
            radius=LIMB_RADIUS * 0.85,
            material=body_material,
        )

    # --------------------------------------------------------
    # Create the head.
    # --------------------------------------------------------

    head = create_joint(
        name="Head",
        position=head_center,
        radius=head_radius,
        material=body_material,
    )

    # Make the head slightly taller than it is wide.
    head.scale = (
        0.88,
        0.82,
        1.08,
    )

    # --------------------------------------------------------
    # Create a small nose marker to indicate which way the
    # mannequin is facing.
    # --------------------------------------------------------

    create_joint(
        name="NoseMarker",
        position=points[NOSE],
        radius=head_radius * 0.13,
        material=joint_material,
    )

    print("Created the pose avatar.")


# ============================================================
# SECTION 12: CREATE A GROUND PLANE
# ============================================================

def create_ground_plane():
    """
    Add a simple floor underneath the mannequin.
    """

    ground_material = create_material(
        name="GroundMaterial",
        color=(0.08, 0.08, 0.08, 1.0),
        roughness=0.9,
    )

    bpy.ops.mesh.primitive_plane_add(
        size=8.0,
        location=(0.0, 0.0, -0.02),
    )

    ground = bpy.context.object
    ground.name = "Ground"

    assign_material(
        ground,
        ground_material,
    )


# ============================================================
# SECTION 13: MAIN PROGRAM
# ============================================================

def main():
    """
    Run all stages in order.
    """

    # Remove Blender's default objects.
    clear_scene()

    # Read the landmark JSON file.
    landmarks = load_landmarks(JSON_PATH)

    # Convert MediaPipe coordinates to Blender coordinates.
    points = build_point_dictionary(landmarks)

    # Move the pose so its feet rest on the ground.
    move_avatar_to_ground(points)

    # Construct the mannequin.
    create_pose_avatar(points)

    # Add a floor underneath it.
    create_ground_plane()

    print("Finished building the Blender pose avatar.")


# Run the program when Blender executes this file.
if __name__ == "__main__":
    main()

# --------------------------------------------------------
# Save the completed Blender scene.
 # --------------------------------------------------------

output_path = (
    Path(__file__).resolve().parent
    / "pose_avatar.blend"
)

bpy.ops.wm.save_as_mainfile(
    filepath=str(output_path)
)

print("Finished building the Blender pose avatar.")
print(f"Saved Blender project to: {output_path}")