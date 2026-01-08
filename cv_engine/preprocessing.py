import cv2

def preprocess_image(image):
    """
    Convert image to enhanced grayscale for texture detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return denoised
