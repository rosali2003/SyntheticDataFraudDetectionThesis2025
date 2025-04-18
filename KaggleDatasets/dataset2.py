import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Download and load the dataset
path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
data = pd.read_csv(path + '/Base.csv')

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Separate features and target
target_column = 'fraud_label'
X = data.drop([target_column], axis=1)
y = data[target_column]

# Convert categorical columns to numeric
categorical_columns = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_columns)

print("\nFeatures after encoding:", X.shape[1])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Prepare DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define PyTorch models
class OneLayerNN(nn.Module):
    def __init__(self):
        super(OneLayerNN, self).__init__()
        self.layer = nn.Linear(X_train.shape[1], 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

class TwoLayerNN(nn.Module):
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 100)
        self.layer2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.sigmoid(self.layer2(x))

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs} completed')

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predictions.extend((outputs > 0.5).numpy().astype(int))
            actuals.extend(labels.numpy().astype(int))

    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    report = classification_report(actuals, predictions)
    cm = confusion_matrix(actuals, predictions)
    tn, fp, fn, tp = cm.ravel()

    print(f'\nAccuracy: {accuracy:.4f}')
    print('\nConfusion Matrix Metrics:')
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}')
    print(f'Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}')
    print('\nClassification Report:')
    print(report)
    print('\nConfusion Matrix:')
    print(cm)

    return accuracy, (tp, tn, fp, fn), report, cm

if __name__ == "__main__":
    print("\nStarting model training and evaluation...")

    # Initialize results dictionary
    results = {}

    # Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_cm = confusion_matrix(y_test, lr_pred)
    tn, fp, fn, tp = lr_cm.ravel()
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred)

    print(f'\nLogistic Regression Results:')
    print(f'Accuracy: {lr_accuracy:.4f}')
    print('\nConfusion Matrix Metrics:')
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}')
    print(f'Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}')
    print('\nClassification Report:')
    print(lr_report)

    results['Logistic Regression'] = (lr_accuracy, (tp, tn, fp, fn), lr_report, lr_cm)

    # Train and evaluate Linear Regression
    print("\nTraining Linear Regression...")
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)
    lin_reg_pred_proba = lin_reg_model.predict(X_test)
    lin_reg_pred = (lin_reg_pred_proba > 0.5).astype(int)
    lin_reg_cm = confusion_matrix(y_test, lin_reg_pred)
    tn, fp, fn, tp = lin_reg_cm.ravel()
    lin_reg_accuracy = accuracy_score(y_test, lin_reg_pred)
    lin_reg_report = classification_report(y_test, lin_reg_pred)

    print(f'\nLinear Regression Results:')
    print(f'Accuracy: {lin_reg_accuracy:.4f}')
    print('\nConfusion Matrix Metrics:')
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}')
    print(f'Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}')
    print('\nClassification Report:')
    print(lin_reg_report)

    results['Linear Regression'] = (lin_reg_accuracy, (tp, tn, fp, fn), lin_reg_report, lin_reg_cm)

    # Train and evaluate Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_cm = confusion_matrix(y_test, rf_pred)
    tn, fp, fn, tp = rf_cm.ravel()
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred)

    print(f'\nRandom Forest Results:')
    print(f'Accuracy: {rf_accuracy:.4f}')
    print('\nConfusion Matrix Metrics:')
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}')
    print(f'Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}')
    print('\nClassification Report:')
    print(rf_report)

    results['Random Forest'] = (rf_accuracy, (tp, tn, fp, fn), rf_report, rf_cm)

    # Create and train neural network models
    print("\nTraining Neural Networks...")
    one_layer_model = OneLayerNN()
    two_layer_model = TwoLayerNN()

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_one = optim.Adam(one_layer_model.parameters())
    optimizer_two = optim.Adam(two_layer_model.parameters())

    # Train models
    print("\nTraining One Layer Neural Network...")
    train_model(one_layer_model, train_loader, criterion, optimizer_one)
    print("\nOne Layer Neural Network Results:")
    results['One Layer NN'] = evaluate_model(one_layer_model, test_loader)

    print("\nTraining Two Layer Neural Network...")
    train_model(two_layer_model, train_loader, criterion, optimizer_two)
    print("\nTwo Layer Neural Network Results:")
    results['Two Layer NN'] = evaluate_model(two_layer_model, test_loader)

    # Compare model performances
    print("\nModel Performance Comparison:")
    print(f"{'Model':<25} {'Accuracy':<10} {'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8}")
    print("-" * 70)
    for model_name, (accuracy, metrics, _, _) in results.items():
        tp, tn, fp, fn = metrics
        print(f"{model_name:<25} {accuracy:.4f}    {tp:<8} {tn:<8} {fp:<8} {fn:<8}")

    # Print feature importance for Random Forest
    print("\nTop 10 Most Important Features (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))

    # Print Linear Regression coefficients
    print("\nTop 10 Most Important Features (Linear Regression):")
    lin_reg_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': np.abs(lin_reg_model.coef_)
    }).sort_values('coefficient', ascending=False)
    print(lin_reg_importance.head(10))

    print("\nAnalysis complete!")
