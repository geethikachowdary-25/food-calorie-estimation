import cv2
import numpy as np

def preprocess_image(image):
    # Resize
    image = cv2.resize(image, (299, 299))

    # Bilateral filtering (edge-preserving)
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # RGB → Lab
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Normalize
    image = image.astype(np.float32) / 255.0

    return image
