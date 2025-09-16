import joblib

# Load the saved model
model = joblib.load("best_model.pkl")

print("✅ Model loaded successfully:", type(model))
