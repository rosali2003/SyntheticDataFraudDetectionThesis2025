import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')  # Add parent directory to path
from volume_dataset import create_dataset

# Create output directory for plots
os.makedirs('fraud_detection_results', exist_ok=True)

# Generate a larger dataset for better training
df = create_dataset(n_samples=10000000, fraud_rate=0.05)  # 5% fraud rate with 1000 samples

# Prepare features
# Convert categorical variables to numerical
le = LabelEncoder()
df['region_encoded'] = le.fit_transform(df['region'])

# Create feature for IP region (first octet)
df['ip_first_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[0]))

# Extract region from billing address
df['billing_region'] = df['billing_address'].apply(lambda x: x.split(',')[1].strip())
df['billing_region_encoded'] = le.fit_transform(df['billing_region'])

# Create features array X
X = pd.DataFrame({
    'ip_first_octet': df['ip_first_octet'],
    'region_encoded': df['region_encoded'],
    'billing_region_encoded': df['billing_region_encoded']
})

# Target variable y
y = df['is_fraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Convert predictions to binary (0 or 1) using 0.5 as threshold
y_pred_binary = (y_pred >= 0.5).astype(int)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print regression metrics with clear formatting
print("\n" + "="*50)
print("REGRESSION METRICS:")
print("="*50)
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"R-squared Score: {r2:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Calculate and print confusion matrix metrics
cm = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*50)
print("CONFUSION MATRIX METRICS:")
print("="*50)
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")

# Calculate and print classification metrics
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

# Print confusion matrix in a more visual format
print("\n" + "="*50)
print("CONFUSION MATRIX VISUALIZATION:")
print("="*50)
print("                  Predicted")
print("                  Negative  Positive")
print("Actual Negative    {:6d}    {:6d}".format(tn, fp))
print("Actual Positive    {:6d}    {:6d}".format(fn, tp))

# Print feature importance
feature_names = X.columns
print("\nFeature Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('fraud_detection_results/actual_vs_predicted.png')
plt.close()

# Plot prediction distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred, bins=50, alpha=0.5, label='Predictions')
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
plt.xlabel('Predicted Values')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions')
plt.legend()
plt.tight_layout()
plt.savefig('fraud_detection_results/prediction_distribution.png')
plt.close()

# Example predictions
print("\nExample Predictions:")
for i in range(min(5, len(X_test))):
    pred = y_pred[i]
    pred_binary = y_pred_binary[i]
    actual = y_test.iloc[i]

    # Get original feature values
    original_features = X_test.iloc[i]

    print(f"\nSample {i+1}:")
    print(f"Features:")
    print(f"  IP First Octet: {original_features['ip_first_octet']}")
    print(f"  IP Region: {le.inverse_transform([int(original_features['region_encoded'])])[0]}")
    print(f"  Billing Region: {le.inverse_transform([int(original_features['billing_region_encoded'])])[0]}")
    print(f"Raw Prediction Score: {pred:.4f}")
    print(f"Binary Prediction: {'Fraud' if pred_binary == 1 else 'Legitimate'}")
    print(f"Actual: {'Fraud' if actual == 1 else 'Legitimate'}")

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('fraud_detection_results/residuals.png')
plt.close()
