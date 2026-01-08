import cv2

def preprocess_image(image):
    """
    Input:
        image: BGR image (NumPy array)
    Output:
        preprocessed grayscale image
    """

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Noise reduction (edge-preserving)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return denoised
