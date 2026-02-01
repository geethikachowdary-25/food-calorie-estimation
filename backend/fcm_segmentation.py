import numpy as np
import cv2

def fcm_segmentation(image, clusters=3, m=2, max_iter=30, epsilon=1e-4):
    # 🔽 Downsample image for speed
    small = cv2.resize(image, (128, 128))
    h, w, c = small.shape

    pixels = small.reshape(-1, c)
    N = pixels.shape[0]

    # Initialize membership matrix
    U = np.random.dirichlet(np.ones(clusters), size=N)

    for _ in range(max_iter):
        U_old = U.copy()

        # Compute cluster centers
        centers = np.array([
            np.sum((U[:, j] ** m).reshape(-1, 1) * pixels, axis=0) /
            np.sum(U[:, j] ** m)
            for j in range(clusters)
        ])

        # Update membership
        for i in range(N):
            for j in range(clusters):
                denom = sum(
                    (np.linalg.norm(pixels[i] - centers[j]) /
                     np.linalg.norm(pixels[i] - centers[k]) + 1e-6) ** (2 / (m - 1))
                    for k in range(clusters)
                )
                U[i, j] = 1 / denom

        if np.linalg.norm(U - U_old) < epsilon:
            break

    labels = np.argmax(U, axis=1)
    segmented_small = centers[labels].reshape(h, w, c)

    # 🔼 Upsample back to original size
    segmented = cv2.resize(segmented_small, (299, 299))
    mask = cv2.resize(labels.reshape(h, w).astype(np.uint8), (299, 299))

    return segmented, mask
