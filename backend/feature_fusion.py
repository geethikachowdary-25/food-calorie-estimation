import numpy as np

def fuse_features(deep_features, segmentation_mask):
    # Portion size = number of segmented pixels
    area = np.sum(segmentation_mask >= 0)

    # Use ONLY ONE portion feature (matches training)
    fused = np.concatenate([deep_features, [area]])

    return fused.reshape(1, -1)
