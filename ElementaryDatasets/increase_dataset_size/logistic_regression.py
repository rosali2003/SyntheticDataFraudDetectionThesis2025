import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')  # Add parent directory to path
from volume_dataset import create_dataset  # Import the dataset generation function

# Create output directory for plots
os.makedirs('fraud_detection_results', exist_ok=True)

# Generate a larger dataset for better training
df = create_dataset(n_samples=10000000, fraud_rate=0.01)  # 1% fraud rate with 10000 samples

# Prepare features
le = LabelEncoder()

# Enhanced feature engineering
# 1. Location-based features
df['region_encoded'] = le.fit_transform(df['region'])
df['ip_first_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[0]))
df['billing_region'] = df['billing_address'].apply(lambda x: x.split(',')[1].strip())
df['billing_region_encoded'] = le.fit_transform(df['billing_region'])

# 2. Location mismatch feature (important for fraud detection)
df['location_mismatch'] = (df['region'] != df['billing_region']).astype(int)

# 3. IP address features
df['ip_second_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[1]))
df['ip_third_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[2]))
df['ip_fourth_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[3]))

# 4. Transaction amount features
df['amount'] = df['transaction_amount']
df['high_amount'] = (df['amount'] > 1000).astype(int)

# Create features array X with enhanced feature set
X = pd.DataFrame({
    'ip_first_octet': df['ip_first_octet'],
    'ip_second_octet': df['ip_second_octet'],
    'ip_third_octet': df['ip_third_octet'],
    'ip_fourth_octet': df['ip_fourth_octet'],
    'region_encoded': df['region_encoded'],
    'billing_region_encoded': df['billing_region_encoded'],
    'location_mismatch': df['location_mismatch'],
    'amount': df['amount'],
    'high_amount': df['high_amount']
})

# Target variable y
y = df['is_fraud']

# Split the data with stratification to maintain fraud ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model with optimized parameters
model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    solver='saga',  # Better for large datasets and L1 penalty
    penalty='l1',   # L1 regularization for feature selection
    C=0.1          # Stronger regularization
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Print detailed results
print("\nModel Evaluation:")
print(f"\nTotal Predictions: {len(y_test)}")
print(f"Total Fraud Cases: {sum(y_test)}")
print(f"Fraud Rate: {(sum(y_test)/len(y_test))*100:.2f}%")

print("\nDetailed Metrics:")
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nDetailed Counts:")
print(f"True Negatives (Correctly identified legitimate): {tn}")
print(f"False Positives (Legitimate marked as fraud): {fp}")
print(f"False Negatives (Fraud marked as legitimate): {fn}")
print(f"True Positives (Correctly identified fraud): {tp}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print feature importance
feature_names = X.columns
print("\nFeature Importance:")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"{name}: {abs(coef):.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('fraud_detection_results/roc_curve.png')
plt.close()

# Plot feature importance
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=True)

plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance in Logistic Regression')
plt.tight_layout()
plt.savefig('fraud_detection_results/feature_importance.png')
plt.close()

# Example predictions with probabilities
print("\nExample Predictions:")
for i in range(min(5, len(X_test))):
    prob = model.predict_proba(X_test_scaled[i:i+1])[0]
    pred = model.predict(X_test_scaled[i:i+1])[0]
    actual = y_test.iloc[i]

    features = X_test.iloc[i]
    print(f"\nSample {i+1}:")
    print(f"Features:")
    for feat_name, feat_val in features.items():
        print(f"  {feat_name}: {feat_val}")
    print(f"Probability of Fraud: {prob[1]:.2%}")
    print(f"Prediction: {'Fraud' if pred == 1 else 'Legitimate'}")
    print(f"Actual: {'Fraud' if actual == 1 else 'Legitimate'}")
