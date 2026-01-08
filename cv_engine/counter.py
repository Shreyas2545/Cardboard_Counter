import numpy as np
from scipy.signal import find_peaks
from utils.config import (
    FLUTES_PER_SHEET,
    PEAK_HEIGHT,
    PEAK_DISTANCE,
    SMOOTHING_WINDOW
)

def count_cardboards(signal):
    """
    Convert corrugation peaks to actual cardboard sheet count
    """

    # Smooth signal
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smooth_signal = np.convolve(signal, kernel, mode="same")

    # Detect peaks (corrugation flutes)
    peaks, _ = find_peaks(
        smooth_signal,
        height=PEAK_HEIGHT,
        distance=PEAK_DISTANCE
    )

    total_flutes = len(peaks)

    # Convert flutes → sheets
    sheet_count = max(1, round(total_flutes / FLUTES_PER_SHEET))

    return sheet_count, total_flutes
