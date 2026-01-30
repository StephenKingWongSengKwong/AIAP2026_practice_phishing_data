import pandas as pd
import joblib
from src.preprocessing import clean_data

# 1. Load the saved model
model = joblib.load('models/final_phishing_model.joblib')

# 2. Simulate new, incoming data (or load a new CSV)
# Note: In a real scenario, this would be a single row of website features
new_data = pd.read_csv('phishing_data.csv').iloc[:5] # Testing on first 5 rows

# 3. Preprocess the new data using the SAME logic as training
X_new, _ = clean_data(new_data)

# 4. Predict
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

for i, pred in enumerate(predictions):
    status = "PHISHING" if pred == 1 else "LEGITIMATE"
    print(f"Site {i+1}: {status} (Confidence: {probabilities[i]:.2f})")