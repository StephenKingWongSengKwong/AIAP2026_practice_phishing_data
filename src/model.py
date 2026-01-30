from sklearn.ensemble import RandomForestClassifier

def get_tuned_rf_model():
    """
    Returns a Random Forest model with the best parameters found 
    during the AIAP assessment tuning phase.
    """
    return RandomForestClassifier(
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# ... keep your existing get_tuned_rf_model and train_model functions ...

def save_model(model, filename='models/final_phishing_model.joblib'):
    """Saves the model to the models directory."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    joblib.dump(model, filename)
    print(f"\n[SUCCESS] Model saved to {filename}")

def load_model(filename='models/final_phishing_model.joblib'):
    """Loads a saved model from disk."""
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        raise FileNotFoundError(f"No model found at {filename}")