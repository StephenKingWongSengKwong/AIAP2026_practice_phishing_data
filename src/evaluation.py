from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def print_model_performance(model, X_test, y_test):
    """
    Prints a detailed classification report and ROC AUC score.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {auc_score:.4f}")
    print("="*30)