# STEP 1: Import the necessary modules
import cv2
import json
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imwrite('annotated_landmarks.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

if detection_result.segmentation_masks:
  segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
  segmentation_mask = np.squeeze(segmentation_mask)

  # Convert to 3-channel uint8 image for visualization.
  visualized_mask = (segmentation_mask * 255).astype(np.uint8)
  visualized_mask = np.stack([visualized_mask] * 3, axis=-1)
  cv2.imwrite('segmentation_mask.jpg', visualized_mask)

print('Wrote annotated_landmarks.jpg')



