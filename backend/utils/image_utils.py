import numpy as np
import cv2

def read_image(file_bytes):
    np_img = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)
