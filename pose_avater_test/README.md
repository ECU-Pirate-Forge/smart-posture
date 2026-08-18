Pose Avatar with MediaPipe and Blender

This project takes a still image, detects body landmarks with MediaPipe, and uses those landmarks to create a simple 3D pose avatar in Blender.

Image
  ↓
MediaPipe + OpenCV
  ↓
3D pose landmarks
  ↓
JSON
  ↓
Blender
  ↓
3D pose avatar

What is Blender?

Blender is a free, open-source 3D creation tool used for modeling, animation, rendering, and Python scripting.

This project uses Blender to turn MediaPipe body coordinates into a 3D mannequin. Blender's Python API, bpy, is used to create and position the avatar objects.

Requirements

Install the following:

pip install mediapipe opencv-python numpy bpy

You also need:

Python 3.13

Blender: https://www.blender.org/download/

MediaPipe Pose Landmarker model:
pose_landmarker_heavy.task

mathutils is included with Blender/bpy, so it does not need to be installed separately.

To test Blender from Python:

python -c "import bpy; from mathutils import Vector; print(bpy.app.version_string)"

For best compatibility, use JPG or PNG images.

mediapipe-landmark-avatar.py

This script handles pose detection.

It:

Loads the input image.

Loads the MediaPipe pose model.

Detects 33 body landmarks.

Draws the detected pose on the image.

Saves the 3D landmark coordinates for Blender.

The important output is:

pose_world_landmarks.json

Each landmark contains values such as:

x
y
z
visibility

Examples of landmark IDs:

11 = Left Shoulder
12 = Right Shoulder
13 = Left Elbow
14 = Right Elbow
23 = Left Hip
24 = Right Hip
25 = Left Knee
26 = Right Knee
27 = Left Ankle
28 = Right Ankle

pose_avater.blend

pose_avater.blend is the Blender project containing the generated 3D avatar.

The avatar is created from simple 3D shapes:

Spheres for joints and the head

Cylinders for arms, legs, and feet

Ellipsoids for the torso and pelvis

The Blender code reads the coordinates from pose_world_landmarks.json and positions each body part to match the detected pose.

MediaPipe and Blender use different coordinate systems, so the coordinates are converted before the avatar is created.

Example:

Vector((
    landmark["x"],
    -landmark["z"],
    -landmark["y"]
))

Running the Project

Run the MediaPipe script first:

python mediapipe-landmark-avater.py

This creates the landmark data.

Then run the Blender avatar script:

python create_pose_avatar.py

This creates or updates:

pose_avater.blend

Open the Blender project on macOS with:

open pose_avater.blend

Using a Different Image

Change the input image used by mediapipe-landmark-avater.py, then run the MediaPipe script again.

The process is:

New image
  ↓
MediaPipe creates new landmarks
  ↓
pose_world_landmarks.json
  ↓
Blender creates the new pose

The Blender script does not read the image directly. It reads the landmark data produced by MediaPipe.

Future Goals

The next stage is to collect more landmarks and make the pose avater more human-like instead of skeletal. It should also 
process video frames, collect landmarks over time, and use Blender keyframes or an armature to animate the avatar.