
import cv2
import numpy as np


def analyze_visibility(image_path):
    """
    Preliminary image-quality heuristic.
    This does NOT reliably distinguish fog from every other
    cause of low contrast, such as darkness or overexposure.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    # Resize for faster processing
    image = cv2.resize(image, (400, 300))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Measure image contrast
    contrast = float(np.std(gray))

    # Measure edge strength
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    # Basic heuristic, not a trained fog classifier
    if contrast < 25 and edge_density < 0.04:
        level = "POOR"
        message = (
            "Low contrast detected. Fog or haze may be present. "
            "Please verify actual road conditions."
        )
    elif contrast < 45 and edge_density < 0.08:
        level = "MODERATE"
        message = (
            "Moderate image contrast detected. "
            "Check visibility before travelling."
        )
    else:
        level = "GOOD"
        message = (
            "The image has relatively clear contrast. "
            "This does not guarantee safe driving visibility."
        )

    return {
        "level": level,
        "contrast": round(contrast, 2),
        "edge_density": round(edge_density, 4),
        "message": message,
    }
