import cv2
import numpy as np

# create a dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("OpenCV working fine!")
print("Image shape:", gray.shape)
