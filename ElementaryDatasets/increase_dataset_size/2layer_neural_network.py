import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')
from volume_dataset import create_dataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for plots
os.makedirs('fraud_detection_results', exist_ok=True)

# Define the neural network
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Generate dataset
df = create_dataset(n_samples=100, fraud_rate=0.05)

# Prepare features
le = LabelEncoder()
df['region_encoded'] = le.fit_transform(df['region'])
df['ip_first_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[0]))
df['billing_region'] = df['billing_address'].apply(lambda x: x.split(',')[1].strip())
df['billing_region_encoded'] = le.fit_transform(df['billing_region'])

# Create additional features
df['region_billing_match'] = (df['region_encoded'] == df['billing_region_encoded']).astype(int)

# Create features array X
X = pd.DataFrame({
    'ip_first_octet': df['ip_first_octet'],
    'region_encoded': df['region_encoded'],
    'billing_region_encoded': df['billing_region_encoded'],
    'region_billing_match': df['region_billing_match']
})

# Target variable y
y = df['is_fraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Initialize the model with improved architecture
model = TwoLayerNet(input_size=X_train.shape[1], hidden_size=64)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization

# Training parameters
num_epochs = 150  # Increased epochs
batch_size = 64   # Increased batch size
n_batches = len(X_train_tensor) // batch_size

# Lists to store metrics
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Early stopping parameters
best_test_loss = float('inf')
patience = 10
patience_counter = 0
best_model_state = None

# Training loop
print("\nTraining Progress:")
print("="*50)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct_train = 0
    total_train = 0

    # Mini-batch training
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_X = X_train_tensor[start_idx:end_idx]
        batch_y = y_train_tensor[start_idx:end_idx]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Calculate training accuracy
        predicted = (outputs >= 0.5).float()
        correct_train += (predicted == batch_y).sum().item()
        total_train += batch_y.size(0)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate average loss and accuracy for the epoch
    avg_train_loss = epoch_loss / n_batches
    train_accuracy = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())

        # Calculate test accuracy
        test_predicted = (test_outputs >= 0.5).float()
        test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        test_accuracies.append(test_accuracy)

    # Early stopping check
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Test Acc: {test_accuracy:.4f}")

    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_tensor).numpy()
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
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

# Plot training and validation metrics
plt.figure(figsize=(12, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('fraud_detection_results/neural_network_metrics.png')
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
plt.savefig('fraud_detection_results/neural_network_roc.png')
plt.close()

# Plot Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(10, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('fraud_detection_results/neural_network_pr_curve.png')
plt.close()

# Example predictions
print("\nExample Predictions:")
for i in range(min(5, len(X_test))):
    pred = y_pred_proba[i][0]
    pred_binary = y_pred_binary[i][0]
    actual = y_test.iloc[i]

    # Get original feature values
    original_features = X_test.iloc[i]

    print(f"\nSample {i+1}:")
    print(f"Features:")
    print(f"  IP First Octet: {original_features['ip_first_octet']}")
    print(f"  IP Region: {le.inverse_transform([int(original_features['region_encoded'])])[0]}")
    print(f"  Billing Region: {le.inverse_transform([int(original_features['billing_region_encoded'])])[0]}")
    print(f"  Region-Billing Match: {'Yes' if original_features['region_billing_match'] else 'No'}")
    print(f"Probability of Fraud: {pred:.4f}")
    print(f"Prediction: {'Fraud' if pred_binary == 1 else 'Legitimate'}")
    print(f"Actual: {'Fraud' if actual == 1 else 'Legitimate'}")

# Save model weights
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
    'label_encoder': le,
    'feature_names': list(X.columns)
}, 'fraud_detection_results/neural_network_checkpoint.pth')
