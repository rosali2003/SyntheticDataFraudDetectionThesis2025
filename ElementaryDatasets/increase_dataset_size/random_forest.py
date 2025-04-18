import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append('..')
from volume_dataset import create_dataset
import joblib

# Create output directory for plots
os.makedirs('fraud_detection_results', exist_ok=True)

# Generate dataset with 1% fraud rate
print("Generating dataset...")
df = create_dataset(n_samples=10000, fraud_rate=0.01)  # 1% fraud rate with 10000 samples

# Verify dataset statistics
fraud_count = df['is_fraud'].sum()
print(f"\nDataset Statistics:")
print(f"Total transactions: {len(df)}")
print(f"Fraud cases: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
print(f"Legitimate cases: {len(df) - fraud_count} ({(1-fraud_count/len(df))*100:.2f}%)")

# Prepare features
print("\nPreparing features...")
le = LabelEncoder()

# Basic features
df['region_encoded'] = le.fit_transform(df['region'])
df['ip_first_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[0]))
df['billing_region'] = df['billing_address'].apply(lambda x: x.split(',')[1].strip())
df['billing_region_encoded'] = le.fit_transform(df['billing_region'])

# Enhanced feature engineering
# 1. Region matching
df['region_billing_match'] = (df['region_encoded'] == df['billing_region_encoded']).astype(int)

# 2. IP address risk scoring
df['ip_second_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[1]))
df['ip_risk_score'] = df.apply(
    lambda row: 1 if row['ip_first_octet'] in [0, 10, 127, 169, 172, 192, 224] else 0,
    axis=1
)

# 3. Geographic distance proxy
df['geo_distance'] = np.abs(df['region_encoded'] - df['billing_region_encoded'])

# Create features array X with enhanced feature set
X = pd.DataFrame({
    'ip_first_octet': df['ip_first_octet'],
    'ip_second_octet': df['ip_second_octet'],
    'region_encoded': df['region_encoded'],
    'billing_region_encoded': df['billing_region_encoded'],
    'region_billing_match': df['region_billing_match'],
    'ip_risk_score': df['ip_risk_score'],
    'geo_distance': df['geo_distance'],
    'transaction_amount': df['transaction_amount'],
    'transaction_count_1h': df['transaction_count_1h'],
    'is_high_risk_amount': df['is_high_risk_amount'],
    'is_suspicious_time': df['is_suspicious_time'],
    'is_velocity_alert': df['is_velocity_alert'],
    'is_business_hour': df['is_business_hour'],
    'transaction_hour': df['transaction_hour']
})

# Target variable y
y = df['is_fraud']

# Split the data without stratification due to small number of fraud cases
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set fraud cases: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
print(f"Test set fraud cases: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 20}]  # Give more weight to fraud cases
}

# Initialize base model
base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    criterion='gini',
    bootstrap=True
)

# Perform grid search with cross-validation
print("\nPerforming grid search with cross-validation...")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,  # Reduced from 5 due to small number of fraud cases
    scoring='average_precision',  # Better metric for imbalanced classification
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Make predictions
print("\nMaking predictions...")
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Print detailed metrics
print("\n" + "="*50)
print("CONFUSION MATRIX METRICS:")
print("="*50)
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")

# Calculate classification metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*50)
print("CLASSIFICATION METRICS:")
print("="*50)
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1 Score: {f1_score:.6f}")

# Print confusion matrix visualization
print("\n" + "="*50)
print("CONFUSION MATRIX VISUALIZATION:")
print("="*50)
print("                  Predicted")
print("                  Negative  Positive")
print("Actual Negative    {:6d}    {:6d}".format(tn, fp))
print("Actual Positive    {:6d}    {:6d}".format(fn, tp))

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('fraud_detection_results/confusion_matrix_heatmap.png')
plt.close()

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
plt.savefig('fraud_detection_results/random_forest_roc.png')
plt.close()

# Plot Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2,
         label=f'PR curve (AUC = {pr_auc:.2f}, AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('fraud_detection_results/random_forest_pr_curve.png')
plt.close()

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('fraud_detection_results/random_forest_feature_importance.png')
plt.close()

# Print feature importance details
print("\n" + "="*50)
print("FEATURE IMPORTANCE RANKING:")
print("="*50)
for idx, row in feature_importance.sort_values('importance', ascending=False).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Example predictions
print("\nExample Predictions:")
for i in range(min(5, len(X_test))):
    pred_prob = y_pred_proba[i]
    pred = y_pred[i]
    actual = y_test.iloc[i]

    # Get original feature values
    original_features = X_test.iloc[i]

    print(f"\nSample {i+1}:")
    print(f"Features:")
    for feature in X.columns:
        print(f"  {feature}: {original_features[feature]}")
    print(f"Probability of Fraud: {pred_prob:.4f}")
    print(f"Prediction: {'Fraud' if pred == 1 else 'Legitimate'}")
    print(f"Actual: {'Fraud' if actual == 1 else 'Legitimate'}")

# Save the model and associated objects
print("\nSaving model and associated objects...")
joblib.dump({
    'model': best_model,
    'scaler': scaler,
    'label_encoder': le,
    'feature_names': list(X.columns),
    'best_params': grid_search.best_params_,
    'cv_results': grid_search.cv_results_,
    'training_metadata': {
        'n_samples': len(df),
        'fraud_rate': fraud_count/len(df),
        'n_features': len(X.columns),
        'best_score': grid_search.best_score_
    }
}, 'fraud_detection_results/random_forest_model.joblib')
