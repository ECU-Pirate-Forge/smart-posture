import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(
    model_asset_path="pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
# Initialize the Pose model
detector = vision.PoseLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
timestamp = 0

while True:
    #check if the webcam is opened correctly
    success, img = cap.read()

    if not success:
        print("Could not read frame from webcam.")
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create an MP image from the RGB frame
    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=img_rgb
)
    results = detector.detect_for_video(
        mp_image,
        timestamp
    )
    timestamp += 33  # Increment by ~33ms for ~30 FPS

    if results.pose_landmarks:

        person = results.pose_landmarks[0] 
         # Assuming single person detection 
        left_shoulder = person

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        print("Left shoulder:", left_shoulder.x, left_shoulder.y)
        print("Right shoulder:", right_shoulder.x, right_shoulder.y)

    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

 