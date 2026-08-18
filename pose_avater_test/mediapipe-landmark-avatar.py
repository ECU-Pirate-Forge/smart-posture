# STEP 1: Import the necessary modules
import cv2
import json
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



def save_world_landmarks(
    detection_result,
    output_path="pose_world_landmarks.json"
):
    """
    Save MediaPipe's 3D pose landmarks to a JSON file.

    Blender will read this JSON file and use the coordinates
    to position the mannequin's joints and limbs.
    """

    # Make sure MediaPipe actually detected a pose.
    if not detection_result.pose_world_landmarks:
        print("No 3D pose landmarks were detected.")
        return False

    # Use the first detected person.
    world_landmarks = detection_result.pose_world_landmarks[0]

    landmarks = []

    # Convert every MediaPipe landmark into regular Python data
    # that can safely be written to JSON.
    for landmark_id, landmark in enumerate(world_landmarks):
        landmarks.append({
            "id": landmark_id,
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
            "visibility": float(landmark.visibility),
            "presence": float(landmark.presence),
        })

    pose_data = {
        "landmark_count": len(landmarks),
        "landmarks": landmarks,
    }



    # Convert the output filename into a Path object.
    output_path = Path(output_path).resolve()

    # Write the landmark information to the JSON file.
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(pose_data, json_file, indent=2)

    print(f"Saved {len(landmarks)} landmarks to:")
    print(output_path)

    return True

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  height, width = annotated_image.shape[:2]

  for pose_landmarks in pose_landmarks_list:
    points = [
      (int(landmark.x * width), int(landmark.y * height))
      for landmark in pose_landmarks
    ]

    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
      start_point = points[connection.start]
      end_point = points[connection.end]
      cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

    for point in points:
      cv2.circle(annotated_image, point, 3, (255, 0, 0), -1)

  return annotated_image


# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='mediapipe-models/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# base_options = python.BaseOptions(model_asset_path='mediapipe-models/efficientdet.tflite')
# options = vision.ObjectDetectorOptions(base_options=base_options,
#                                        score_threshold=0.5)
# detector = vision.ObjectDetector.create_from_options(options)


# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")


# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# Save the raw 3D coordinates for Blender.
save_world_landmarks(
    detection_result,
    "pose_world_landmarks.json"
)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imwrite('annotated_landmarks.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))



