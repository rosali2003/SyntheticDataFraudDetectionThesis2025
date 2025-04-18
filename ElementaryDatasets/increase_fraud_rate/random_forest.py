import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import sys
import os

from fraud_rate import create_dataset

def train_and_evaluate(fraud_rate=0.25, n_samples=1000, n_estimators=100, max_depth=None):
    """
    Train Random Forest model and evaluate performance metrics

    Args:
        fraud_rate (float): Percentage of fraudulent transactions
        n_samples (int): Total number of samples to generate
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees (None for unlimited)
    """
    # Set random seed for reproducibility
    np.random.seed(42)

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
    print(f"Total samples: {len(df)}")

    # Calculate desired number of samples for each class
    n_fraud = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_fraud

    # Separate and resample classes to achieve desired numbers
    df_fraud = df[df['is_fraud'] == 1]
    df_legitimate = df[df['is_fraud'] == 0]

    if len(df_fraud) > 0:
        df_fraud_resampled = resample(df_fraud,
                                    replace=True,
                                    n_samples=n_fraud,
                                    random_state=42)
        df_legitimate_resampled = resample(df_legitimate,
                                         replace=True,
                                         n_samples=n_legitimate,
                                         random_state=42)

        # Combine resampled classes
        df_balanced = pd.concat([df_legitimate_resampled, df_fraud_resampled])

        print("\nClass distribution after resampling:")
        print(df_balanced['is_fraud'].value_counts())
        print(f"Total samples: {len(df_balanced)}")
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

    # Initialize and train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Train on full training set
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print("\nModel Performance Metrics:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Random Forest specific metrics
    print("\nRandom Forest Metrics:")
    print(f"Number of trees: {n_estimators}")
    print(f"Max depth: {max_depth if max_depth else 'unlimited'}")

    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(features, model.feature_importances_):
        print(f"{feature}: {importance:.3f}")

    # Tree structure information
    print("\nTree Information:")
    n_nodes = [tree.tree_.node_count for tree in model.estimators_]
    max_depth_actual = [tree.tree_.max_depth for tree in model.estimators_]
    print(f"Average number of nodes per tree: {np.mean(n_nodes):.1f}")
    print(f"Average actual depth: {np.mean(max_depth_actual):.1f}")

    # Sample predictions with probabilities
    print("\nSample Predictions (first 5 test samples):")
    for i in range(min(5, len(y_test))):
        print(f"True: {y_test[i]}, Predicted: {y_pred[i]}, Probability: {y_pred_proba[i]:.3f}")

    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'cv_score_mean': float(cv_scores.mean()),
        'cv_score_std': float(cv_scores.std()),
        'feature_importance': dict(zip(features, model.feature_importances_.tolist()))
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
