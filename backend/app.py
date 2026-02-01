from fastapi import FastAPI, File, UploadFile
from .utils.image_utils import read_image
from .preprocessing import preprocess_image
from .fcm_segmentation import fcm_segmentation
from .feature_extraction import extract_features
from .feature_fusion import fuse_features
from .calorie_regressor import predict_calories

app = FastAPI(title="Food Calorie Estimation System")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("➡️ Reading image")
        image = read_image(await file.read())

        print("➡️ Preprocessing")
        preprocessed = preprocess_image(image)

        print("➡️ FCM segmentation")
        segmented, mask = fcm_segmentation(preprocessed)

        print("➡️ Feature extraction (InceptionV3)")
        deep_features = extract_features(segmented)

        print("➡️ Feature fusion")
        fused = fuse_features(deep_features, mask)

        print("➡️ Predicting calories")
        calories = predict_calories(fused)

        print("✅ Done")
        return {"estimated_calories": round(calories, 2)}

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

