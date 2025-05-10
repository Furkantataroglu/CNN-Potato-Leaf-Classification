from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from typing import List

app = FastAPI()

# Load the model once at startup
MODEL = tf.keras.models.load_model(
    "/Users/furkantataroglu/Desktop/cnn potato/models/2.keras"
)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Convert raw image bytes into an RGB NumPy array.
    """
    return np.array(Image.open(BytesIO(data)).convert("RGB"))

@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...)
):
    """
    Batch prediction endpoint.

    Accepts multiple image files under the same form field.
    For each image:
      1. Read file bytes.
      2. Decode to RGB array.
      3. Expand dims to shape (1, H, W, C).
      4. Run inference with the loaded model.
      5. Record filename, predicted class, and confidence.
    Returns:
      A JSON object containing a list of prediction results.
    """
    results = []

    for file in files:
        # 1. Read raw bytes
        contents = await file.read()

        # 2. Convert to image array
        image = read_file_as_image(contents)

        # 3. Prepare batch dimension
        img_batch = np.expand_dims(image, 0)

        # 4. Run model inference
        preds = MODEL.predict(img_batch)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))

        # 5. Append result for this file
        results.append({
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[idx],
            "confidence": round(conf * 100, 2)
        })

    return {"predictions": results}
