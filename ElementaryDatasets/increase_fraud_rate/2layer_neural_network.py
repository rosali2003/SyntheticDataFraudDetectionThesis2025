import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sys
import os

from fraud_rate import create_dataset

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size=8):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

def train_and_evaluate(fraud_rate=0.25, n_samples=1000, epochs=100, learning_rate=0.01, hidden_size=8):
    """
    Train two-layer neural network model and evaluate performance metrics

    Args:
        fraud_rate (float): Percentage of fraudulent transactions
        n_samples (int): Total number of samples to generate
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        hidden_size (int): Number of neurons in the hidden layer
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate dataset
    print(f"\nGenerating dataset with fraud_rate={fraud_rate} and {n_samples} samples...")
    df = create_dataset(n_samples=n_samples, fraud_rate=fraud_rate)

    # Print available columns
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())

    # Extract region information and create features
    df['ip_region'] = df['ip_address'].apply(lambda x: 'Asia' if '223' in x or '218' in x or '202' in x or '211' in x
                                           else 'Europe' if '176' in x or '151' in x
                                           else 'North America')
    df['billing_region'] = df['region']
    df['region_mismatch'] = (df['ip_region'] != df['billing_region']).astype(int)

    # Print class distribution before balancing
    print("\nClass distribution before balancing:")
    print(df['is_fraud'].value_counts())

    # Separate majority and minority classes
    df_majority = df[df['is_fraud'] == 0]
    df_minority = df[df['is_fraud'] == 1]

    # Upsample minority class
    if len(df_minority) > 0:
        df_minority_upsampled = resample(df_minority,
                                       replace=True,
                                       n_samples=len(df_majority),
                                       random_state=42)

        # Combine majority class with upsampled minority class
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        print("\nClass distribution after balancing:")
        print(df_balanced['is_fraud'].value_counts())
    else:
        df_balanced = df
        print("\nWarning: No fraud cases found in the dataset!")

    # Prepare features
    features = ['region_mismatch']
    X = df_balanced[features].values
    y = df_balanced['is_fraud'].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # Initialize model
    model = TwoLayerNet(input_size=len(features), hidden_size=hidden_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print("\nTraining neural network...")
    losses = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        losses.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred_raw = model(X_test_tensor)
        y_pred = (y_pred_raw >= 0.5).float()

    # Calculate metrics
    tn = sum((y_test == 0) & (y_pred.numpy().flatten() == 0))
    fp = sum((y_test == 0) & (y_pred.numpy().flatten() == 1))
    fn = sum((y_test == 1) & (y_pred.numpy().flatten() == 0))
    tp = sum((y_test == 1) & (y_pred.numpy().flatten() == 1))

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Print results
    print("\nModel Performance Metrics:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {accuracy:.3f}")

    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Neural network specific metrics
    final_loss = criterion(y_pred_raw, torch.FloatTensor(y_test).reshape(-1, 1)).item()
    print(f"\nNeural Network Metrics:")
    print(f"Final Loss (BCE): {final_loss:.3f}")
    print(f"Average Training Loss: {sum(losses[-10:]) / 10:.3f}")

    # Model architecture and parameters
    print("\nModel Architecture:")
    print(f"Input Size: {len(features)}")
    print(f"Hidden Layer Size: {hidden_size}")
    print(f"Output Size: 1")

    print("\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for name, param in model.named_parameters():
        print(f"\n{name}:")
        print(param.data.numpy())

    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'final_loss': float(final_loss),
        'avg_train_loss': float(sum(losses[-10:]) / 10)
    }

if __name__ == "__main__":
    # Test with different fraud rates
    fraud_rates = [0.01, 0.05, 0.10]
    results = {}

    for rate in fraud_rates:
        print(f"\n{'='*50}")
        print(f"Testing with fraud rate: {rate*100}%")
        results[rate] = train_and_evaluate(fraud_rate=rate)

    # Compare results across fraud rates
    print("\nComparison across fraud rates:")
    metrics_df = pd.DataFrame(results).T
    metrics_df.index = [f"{rate*100}%" for rate in fraud_rates]
    print(metrics_df)
