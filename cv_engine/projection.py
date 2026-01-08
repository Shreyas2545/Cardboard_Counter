import numpy as np

def project_to_1d(gradient_image, direction="vertical"):
    """
    Convert 2D texture image to 1D periodic signal
    """
    if direction == "vertical":
        signal = np.sum(gradient_image, axis=1)
    else:
        signal = np.sum(gradient_image, axis=0)

    signal = signal / (np.max(signal) + 1e-6)
    return signal
