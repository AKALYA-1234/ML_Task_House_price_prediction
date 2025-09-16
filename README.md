🏡 HOUSE PRICE PREDICTION – MLOps PROJECT

Damo app link:https://huggingface.co/spaces/AKALYAS/House_price_prediction
This project demonstrates a simple MLOps pipeline for predicting house prices using the King County House Dataset.
It converts a Jupyter notebook into a reproducible, production-ready workflow with training, prediction, and deployment components.


📂 PROJECT STRUCTURE

📁 project/
│
├── 📁 data/ – 📄 Raw / processed data (optional)
├── 📁 src/
│ ├── 📝 train.py – 🏋️‍♂️ Train the model
│ ├── 📝 predict.py – 🔮 Load the trained model and make predictions
│
├── 📁 artifacts/ – 💾 Saved model artifacts (created after training)
├── 📄 kc_house_data.csv – 🏠 Dataset
├── 📄 requirements.txt – 📦 Dependencies
├── ⚙️ config.yaml (optional) – 📝 Hyperparameters and paths
└── 📄 README.txt – 📖 This file



⚙️ SETUP INSTRUCTIONS

1️⃣ Clone the repository
💻 git clone https://github.com/yourusername/house-price-mlops.git
💻 cd house-price-mlops

2️⃣ Create and activate a virtual environment (recommended)

Windows: python -m venv venv && venv\Scripts\activate

Linux/Mac: python3 -m venv venv && source venv/bin/activate

3️⃣ Install dependencies
💻 pip install -r requirements.txt




🏋️ TRAINING THE MODEL

Run the training script. This loads kc_house_data.csv, trains a model, and saves it in artifacts/model.pkl:

💻 python src/train.py

If successful, you’ll see:
✅ “Model trained and saved to artifacts/model.pkl”




🔮 MAKING PREDICTIONS

Use the saved model to make predictions with new data:

💻 python src/predict.py

The script loads the model from artifacts/model.pkl and prints the predicted house price.



🚀 NEXT STEPS

🔄 Add MLflow for experiment tracking

📦 Add DVC for data versioning

🐳 Create a Dockerfile to containerize training and inference

🤖 Use GitHub Actions or GitLab CI for continuous integration / deployment

🌐 Deploy the model with FastAPI or Flask for a REST API


visualizations:

Correlation Heatmap of Features:
<img width="1136" height="926" alt="image" src="https://github.com/user-attachments/assets/197f9959-d311-4d80-a94e-c754d2420b7f" />



House Prices by Geographic Location (lat, long):
<img width="1010" height="547" alt="image" src="https://github.com/user-attachments/assets/8e4929d3-d7dc-48f3-baa5-e13e0d2c3696" />



Average Price vs Bedrooms:
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/357042e7-4d8d-4da0-9261-31c84bedd019" />




Average Price vs Bathrooms:
<img width="833" height="470" alt="image" src="https://github.com/user-attachments/assets/39f6eac4-6949-4598-b46a-db3c42f562b2" />



verage Price vs Floors:
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/7e9f6935-f53a-4331-947a-b28b1b265d23" />



Average Price vs Condition:
<img width="876" height="470" alt="image" src="https://github.com/user-attachments/assets/db73725f-993b-4449-b045-954e93088536" />



Average Price vs Grade:
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/f48a6242-fab3-4407-b8ab-d9e3a9035ecc" />



Average Price vs View:
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/e3c3f10a-bdc2-4818-9c5f-0557ec1dee3f" />



Average Price: Waterfront vs Non-Waterfront:
<img width="536" height="470" alt="image" src="https://github.com/user-attachments/assets/97fd3ab3-c092-472c-825d-1bea894d3441" />


Demo for App:
<img width="1662" height="733" alt="image" src="https://github.com/user-attachments/assets/4fa163b2-5424-4434-8f16-a573532d10c9" />
