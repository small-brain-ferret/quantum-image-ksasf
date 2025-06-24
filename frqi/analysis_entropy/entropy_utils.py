import numpy as np
from scipy.stats import entropy

def image_entropy(image: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a grayscale image (pixel intensities 0-255).
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255), density=True)
    # Remove zero entries for log2
    hist = hist[hist > 0]
    return entropy(hist, base=2)
