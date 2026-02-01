import os
import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# -----------------------------
# Load InceptionV3 (feature extractor)
# -----------------------------
base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    features = model.predict(img, verbose=0)
    return features.flatten()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset/labels.csv")

X = []
y = []

for _, row in df.iterrows():
    img_path = os.path.join("dataset", row["image_path"])
    features = extract_features(img_path)

    # Simple portion cue (image area proxy)
    img = cv2.imread(img_path)
    area = img.shape[0] * img.shape[1]

    fused_features = np.append(features, area)

    X.append(fused_features)
    y.append(row["calories"])

X = np.array(X)
y = np.array(y)

# -----------------------------
# Train Random Forest
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
os.makedirs("backend/models", exist_ok=True)
joblib.dump(rf, "backend/models/rf_calorie_model.pkl")

print("✅ Random Forest trained and saved successfully")
