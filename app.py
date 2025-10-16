# app.py
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ✅ Load dataset and train model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "model.pkl")  # Save model

# ✅ Load model from code (open-source style)
model = joblib.load("model.pkl")

# ✅ FastAPI app
app = FastAPI()

# ✅ Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ✅ Inference endpoint with logging
@app.post("/predict")
async def predict(data: IrisInput, request: Request):
    input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_array)[0]
    logging.info(f"Request from {request.client.host} | Input: {data.dict()} | Prediction: {prediction}")
    return {"prediction": iris.target_names[prediction]}

# ✅ Run locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
