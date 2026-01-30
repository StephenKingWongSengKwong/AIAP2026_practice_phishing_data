import pandas as pd
from src.preprocessing import clean_data
from sklearn.model_selection import train_test_split
from src.model import get_tuned_rf_model

# Load and clean
df = pd.read_csv('data/phishing_data.csv')
df_clean = clean_data(df)

# Check the label type and values
print(f"Label column unique values: {df_clean['label'].unique()}")
print(f"Label column distribution:\n{df_clean['label'].value_counts()}")

# Split
X = df_clean.drop('label', axis=1)
y = df_clean['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = get_tuned_rf_model()
model.fit(X_train, y_train)

# Check mapping
print(f"Model Classes: {model.classes_}")

# Test a few manual predictions
preds = model.predict(X_test[:10])
actual = y_test[:10].values
print(f"Predictions: {preds}")
print(f"Actual:      {actual}")