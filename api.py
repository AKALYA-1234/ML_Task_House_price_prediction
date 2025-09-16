from fastapi import FastAPI
import joblib
import pandas as pd

# Create FastAPI app
app = FastAPI()

# Load your trained model
model = joblib.load("best_model.pkl")

# Homepage (optional)
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}
