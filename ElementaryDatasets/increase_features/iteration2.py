import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

def generate_dataset(n_samples=10000, fraud_rate=0.01):
    """
    Generate dataset with IP address, billing region, and shipping region features.
    Fraud is determined by:
    1. IP region doesn't match billing region
    2. Shipping region doesn't match billing region
    """
    # Define regions and their corresponding IP ranges
    regions = {
        'Asia': ['223', '218', '202', '211'],
        'Europe': ['176', '151'],
        'North America': ['98', '104']
    }

    # Generate random data
    n_fraud = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_fraud

    # Generate legitimate transactions (matching regions)
    legitimate_data = []
    for _ in range(n_legitimate):
        # Pick a random region
        region = np.random.choice(list(regions.keys()))
        # Generate matching IP and regions
        ip_prefix = np.random.choice(regions[region])
        ip_suffix = np.random.randint(0, 256)
        ip_address = f"{ip_prefix}.{ip_suffix}.0.0"
        legitimate_data.append({
            'ip_address': ip_address,
            'billing_region': region,
            'shipping_region': region,  # Same as billing for legitimate transactions
            'is_fraud': 0
        })

    # Generate fraudulent transactions (mismatched regions)
    fraud_data = []
    for _ in range(n_fraud):
        # Randomly choose type of fraud:
        # 1. IP region mismatch with matching shipping
        # 2. Shipping region mismatch with matching IP
        # 3. All regions different
        fraud_type = np.random.choice(['ip', 'shipping', 'both'])

        if fraud_type == 'ip':
            # Different IP region but same shipping region
            region1, region2 = np.random.choice(list(regions.keys()), size=2, replace=False)
            ip_prefix = np.random.choice(regions[region1])
            ip_suffix = np.random.randint(0, 256)
            ip_address = f"{ip_prefix}.{ip_suffix}.0.0"
            fraud_data.append({
                'ip_address': ip_address,
                'billing_region': region2,
                'shipping_region': region2,
                'is_fraud': 1
            })
        elif fraud_type == 'shipping':
            # Same IP region but different shipping region
            region = np.random.choice(list(regions.keys()))
            ip_prefix = np.random.choice(regions[region])
            ip_suffix = np.random.randint(0, 256)
            ip_address = f"{ip_prefix}.{ip_suffix}.0.0"
            shipping_region = np.random.choice([r for r in regions.keys() if r != region])
            fraud_data.append({
                'ip_address': ip_address,
                'billing_region': region,
                'shipping_region': shipping_region,
                'is_fraud': 1
            })
        else:  # both
            # All regions different
            all_regions = list(regions.keys())
            np.random.shuffle(all_regions)
            ip_region, billing_region, shipping_region = all_regions[:3]
            ip_prefix = np.random.choice(regions[ip_region])
            ip_suffix = np.random.randint(0, 256)
            ip_address = f"{ip_prefix}.{ip_suffix}.0.0"
            fraud_data.append({
                'ip_address': ip_address,
                'billing_region': billing_region,
                'shipping_region': shipping_region,
                'is_fraud': 1
            })

    # Combine and shuffle data
    df = pd.DataFrame(legitimate_data + fraud_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

class SingleLayerNet(nn.Module):
    def __init__(self, input_size):
        super(SingleLayerNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size=8):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

def prepare_data(df, n_samples=10000, fraud_rate=0.01):
    """Prepare dataset for training"""
    # Extract region from IP address
    df['ip_region'] = df['ip_address'].apply(lambda x: 'Asia' if any(prefix in x for prefix in ['223', '218', '202', '211'])
                                           else 'Europe' if any(prefix in x for prefix in ['176', '151'])
                                           else 'North America')

    # Create feature columns
    df['ip_region_mismatch'] = (df['ip_region'] != df['billing_region']).astype(int)
    df['shipping_region_mismatch'] = (df['shipping_region'] != df['billing_region']).astype(int)

    # Prepare features and target
    X = df[['ip_region_mismatch', 'shipping_region_mismatch']].values
    y = df['is_fraud'].values

    return X, y

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print evaluation metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Confusion Matrix (out of {len(y_true)} samples):")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Total test samples: {tp + fp + tn + fn}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return {
        'model': model_name,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

def train_and_evaluate_all_models(n_samples=1000, fraud_rate=0.01):
    """Train and evaluate all models"""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate dataset
    print(f"\nGenerating dataset with {n_samples} samples and {fraud_rate*100}% fraud rate...")
    df = generate_dataset(n_samples=n_samples, fraud_rate=fraud_rate)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print("Total Samples:", len(df))
    print("Fraud Distribution:")
    print(df['is_fraud'].value_counts())
    print("\nMismatch Types:")
    df['ip_region'] = df['ip_address'].apply(lambda x: 'Asia' if any(prefix in x for prefix in ['223', '218', '202', '211'])
                                           else 'Europe' if any(prefix in x for prefix in ['176', '151'])
                                           else 'North America')
    print("IP-Billing Region Mismatches:", sum(df['ip_region'] != df['billing_region']))
    print("Shipping-Billing Region Mismatches:", sum(df['shipping_region'] != df['billing_region']))

    # Prepare data
    X, y = prepare_data(df, n_samples=n_samples, fraud_rate=fraud_rate)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log = log_reg.predict(X_test_scaled)
    results.append(evaluate_model(y_test, y_pred_log, "Logistic Regression"))

    # 2. Linear Regression (with threshold)
    print("\nTraining Linear Regression...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    y_pred_lin = (lin_reg.predict(X_test_scaled) >= 0.5).astype(int)
    results.append(evaluate_model(y_test, y_pred_lin, "Linear Regression"))

    # 3. Single Layer Neural Network
    print("\nTraining Single Layer Neural Network...")
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    single_layer = SingleLayerNet(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(single_layer.parameters(), lr=0.01)

    # Train
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = single_layer(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    single_layer.eval()
    with torch.no_grad():
        y_pred_single = (single_layer(X_test_tensor) >= 0.5).float().numpy()
    results.append(evaluate_model(y_test, y_pred_single, "Single Layer Neural Network"))

    # 4. Two Layer Neural Network
    print("\nTraining Two Layer Neural Network...")
    two_layer = TwoLayerNet(input_size=X.shape[1])
    optimizer = torch.optim.Adam(two_layer.parameters(), lr=0.01)

    # Train
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = two_layer(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    two_layer.eval()
    with torch.no_grad():
        y_pred_two = (two_layer(X_test_tensor) >= 0.5).float().numpy()
    results.append(evaluate_model(y_test, y_pred_two, "Two Layer Neural Network"))

    # 5. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    # Print feature importance for Random Forest
    print("\nRandom Forest Feature Importance:")
    for feature, importance in zip(['ip_region_mismatch', 'shipping_region_mismatch'], rf.feature_importances_):
        print(f"{feature}: {importance:.3f}")

    results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))

    # Compare all models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame(results)
    comparison_df.set_index('model', inplace=True)
    print("\nMetrics for all models:")
    print(comparison_df)

    return comparison_df

if __name__ == "__main__":
    # Test with different fraud rates
    fraud_rates = [0.01, 0.05, 0.10]
    all_results = {}

    for rate in fraud_rates:
        print(f"\n{'='*50}")
        print(f"Testing with fraud rate: {rate*100}%")
        all_results[rate] = train_and_evaluate_all_models(fraud_rate=rate)

    # Save results to CSV
    for rate, df in all_results.items():
        df.to_csv(f'model_comparison_fraud_rate_{rate}_features_2.csv')
