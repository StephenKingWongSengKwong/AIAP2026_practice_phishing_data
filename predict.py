import pandas as pd
import joblib
from src.preprocessing import clean_data

# 1. Load the saved model
model = joblib.load('models/final_phishing_model.joblib')

# 2. Load the sample data
new_data = pd.read_csv('data/phishing_data.csv').iloc[:5]

# 3. Preprocess
X_new, _ = clean_data(new_data)

# 4. FIX: Align columns with the model's expected features
# Get the list of features the model was trained on
model_features = model.feature_names_in_

# Add missing columns with 0s and remove extra columns
X_new = X_new.reindex(columns=model_features, fill_value=0)

# 5. Predict
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

print("\n--- Prediction Results ---")
for i, pred in enumerate(predictions):
    status = "PHISHING" if pred == 1 else "LEGITIMATE"
    print(f"Site {i+1}: {status} (Confidence: {probabilities[i]:.2f})")