import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC  # Faster than SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')


from data_resample import X_train_res, y_train_res, X_test_scaled, y_test



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    print(f"\n{'='*40}")
    print(f"{type(model).__name__} Performance")
    print('='*40)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # ROC Curve
    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{type(model).__name__} (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Linear SVM": LinearSVC(class_weight='balanced', random_state=42)  # Faster than SVC
}

# Train and evaluate
for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Training {name}...")
    
    # Special handling for LinearSVC (no predict_proba)
    if name == "Linear SVM":
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_score = model.decision_function(X_test_scaled)
        
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # ROC for LinearSVC
        roc_auc = roc_auc_score(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f'LinearSVC (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.legend()
        plt.show()
    else:
        model.fit(X_train_res, y_train_res)
        evaluate_model(model, X_test_scaled, y_test)


# Compare model performance

results = []
for name, model in models.items():
    if name == "Linear SVM":
        y_pred = model.predict(X_test_scaled)
        y_score = model.decision_function(X_test_scaled)
        roc_auc = roc_auc_score(y_test, y_score)
    else:
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
        roc_auc = roc_auc_score(y_test, y_prob)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Fraud)': report['1']['precision'],
        'Recall (Fraud)': report['1']['recall'],
        'F1-Score (Fraud)': report['1']['f1-score'],
        'ROC AUC': roc_auc
    })

results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df.round(3))


#EXAMPLE TRANSCATION

def detect_fraud(new_transaction, model, scaler):
    """Predict fraud probability for a new transaction"""
    # Convert to DataFrame and scale
    transaction_df = pd.DataFrame([new_transaction])
    scaled_data = scaler.transform(transaction_df)
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled_data)[0][1]
    else:  # For LinearSVC
        proba = 1/(1+np.exp(-model.decision_function(scaled_data)[0]))
    
    return proba > 0.5, proba  # (is_fraud, probability)

# Example usage:
sample_tx = {
    'Time': 406, 'Amount': 220.00,
    'V1': -1.23, 'V2': 0.45, 'V3': -0.78, 'V4': 2.1, 'V5': -0.5,
    'V6': -1.2, 'V7': -0.3, 'V8': 0.1, 'V9': -0.4, 'V10': 0.3,
    'V11': -2.5, 'V12': 0.9, 'V13': -0.7, 'V14': -1.9, 'V15': 0.2,
    'V16': -0.3, 'V17': -1.8, 'V18': 0.7, 'V19': -0.2, 'V20': 0.1,
    'V21': -0.4, 'V22': 0.3, 'V23': -0.1, 'V24': 0.2, 'V25': -0.3,
    'V26': 0.1, 'V27': -0.2, 'V28': 0.01
}

is_fraud, prob = detect_fraud(sample_tx, models["Logistic Regression"], scaler)
print(f"Fraudulent: {is_fraud} (Probability: {prob:.2%})")