import cv2
import numpy as np

def compute_directional_gradient(image, direction="vertical"):
    """
    Extract texture gradients caused by corrugation layers
    """
    if direction == "vertical":
        grad = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    else:
        grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    magnitude = np.abs(grad)
    magnitude = magnitude / (np.max(magnitude) + 1e-6)

    return magnitude
