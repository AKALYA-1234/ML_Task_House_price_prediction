import gradio as gr
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Prediction function
def predict_price(bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
                  view, condition, grade, sqft_above, sqft_basement, yr_built,
                  yr_renovated, lat, long, sqft_living15, sqft_lot15):

    features = [[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
                 view, condition, grade, sqft_above, sqft_basement, yr_built,
                 yr_renovated, lat, long, sqft_living15, sqft_lot15]]

    prediction = model.predict(features)[0]
    return f"Predicted House Price: ${prediction:,.2f}"

# Gradio UI
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Sqft Living"),
        gr.Number(label="Sqft Lot"),
        gr.Number(label="Floors"),
        gr.Number(label="Waterfront (0/1)"),
        gr.Number(label="View (0-4)"),
        gr.Number(label="Condition (1-5)"),
        gr.Number(label="Grade (1-13)"),
        gr.Number(label="Sqft Above"),
        gr.Number(label="Sqft Basement"),
        gr.Number(label="Year Built"),
        gr.Number(label="Year Renovated"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
        gr.Number(label="Sqft Living 15"),
        gr.Number(label="Sqft Lot 15"),
    ],
    outputs="text",
    title="🏡 House Price Prediction App",
)

demo.launch()
