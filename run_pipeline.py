import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_data
from src.model import get_tuned_rf_model, train_model
from src.evaluation import print_model_performance

# 1. Load Data
# Use low_memory=False to prevent DtypeWarnings which can mess with encoding
df_raw = pd.read_csv('data/phishing_data.csv', low_memory=False)

# 2. Clean - Unpack X and y separately
X, y = clean_data(df_raw)

# 3. Split - Standard 80/20 split
# Using random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train
print("Training Random Forest...")
model = get_tuned_rf_model()
train_model(model, X_train, y_train)

# 5. Evaluate
print_model_performance(model, X_test, y_test)

from src.model import save_model
save_model(model, 'models/final_phishing_model.joblib')
