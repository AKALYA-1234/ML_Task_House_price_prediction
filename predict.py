import joblib
import pandas as pd

# Load trained model
model = joblib.load("best_model.pkl")

# Load dataset
df = pd.read_csv("kc_house_data.csv")

# Drop unused columns
df = df.drop(columns=["id", "date", "zipcode"])

# Separate features
X = df.drop("price", axis=1)

# Make predictions for first 5 houses
preds = model.predict(X.head())
print("Sample predictions:", preds)
