import numpy as np
import cv2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

model = InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_features(image_lab):
    # Lab → RGB
    image_rgb = cv2.cvtColor(
        (image_lab * 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB
    )
    image_rgb = cv2.resize(image_rgb, (299, 299))

    # Prepare for CNN
    image_rgb = image_rgb.astype(np.float32)
    image_rgb = np.expand_dims(image_rgb, axis=0)

    image_rgb = preprocess_input(image_rgb)

    features = model.predict(image_rgb)
    return features.flatten()
