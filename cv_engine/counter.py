import numpy as np
from scipy.signal import find_peaks

def estimate_sheet_count_with_confidence(signal):
    """
    Physics-aware estimation with confidence score.
    """

    smooth = np.convolve(signal, np.ones(7) / 7, mode="same")

    peaks, _ = find_peaks(
        smooth,
        height=0.15,
        distance=3
    )

    if len(peaks) < 2:
        return 1, 0.95

    spacings = np.diff(peaks)
    avg_spacing = np.mean(spacings)
    spacing_std = np.std(spacings)

    stack_thickness = len(signal)
    raw_estimate = stack_thickness / avg_spacing

    SINGLE_SHEET_THRESHOLD = 30

    if raw_estimate < SINGLE_SHEET_THRESHOLD:
        final_count = 1
    else:
        final_count = int(round(raw_estimate))

    # ---- CONFIDENCE CALCULATION ----
    # More uniform spacing = higher confidence
    if avg_spacing == 0:
        confidence = 0.6
    else:
        confidence = 1 - (spacing_std / avg_spacing)

    confidence = max(0.6, min(confidence, 0.98))
    confidence = round(confidence, 2)

    return final_count, confidence
