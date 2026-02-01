import joblib
import numpy as np

MODEL_PATH = "backend/models/rf_calorie_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_calories(fused_features):
    fused_features = np.array(fused_features).reshape(1, -1)
    return float(model.predict(fused_features)[0])
