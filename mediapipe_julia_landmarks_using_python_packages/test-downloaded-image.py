
IMAGE_FILE = 'image.jpg'

import cv2

img = cv2.imread(IMAGE_FILE)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()