import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample
import sys
import os

from fraud_rate import create_dataset

def train_and_evaluate(fraud_rate=0.25, n_samples=1000):
    """
    Train logistic regression model and evaluate performance metrics

    Args:
        fraud_rate (float): Percentage of fraudulent transactions
        n_samples (int): Total number of samples to generate
    """
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
    X = df_balanced[features]
    y = df_balanced['is_fraud']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with class weight adjustment
    print("\nTraining logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)

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

    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(features, model.coef_[0]):
        print(f"{feature}: {importance:.3f}")

    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
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
