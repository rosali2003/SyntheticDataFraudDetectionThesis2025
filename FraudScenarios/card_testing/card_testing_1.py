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
import os

class SingleLayerNet(nn.Module):
    """Single layer neural network with sigmoid activation"""
    def __init__(self, input_size):
        super(SingleLayerNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.layer[0].weight)

    def forward(self, x):
        return self.layer(x)

class TwoLayerNet(nn.Module):
    """Two layer neural network with ReLU and sigmoid activations"""
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

def luhn_checksum(card_number):
    """
    Validate credit card number using Luhn algorithm
    Returns True if valid, False if invalid
    """
    def digits_of(n):
        return [int(d) for d in str(n)]

    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d * 2))
    return checksum % 10 == 0

def generate_card_number(is_valid=True):
    """
    Generate a credit card number
    is_valid: if True, generates valid number, if False, generates invalid number
    """
    # Generate first 15 digits randomly
    prefix = np.random.choice(['4', '5'])  # Visa or Mastercard
    body = ''.join([str(np.random.randint(0, 10)) for _ in range(14)])
    base_number = prefix + body

    if is_valid:
        # Calculate valid check digit
        digits = [int(d) for d in base_number]
        odd_sum = sum(digits[-2::-2])
        even_sum = sum(sum(divmod(2 * d, 10)) for d in digits[-1::-2])
        checksum = (10 - (odd_sum + even_sum) % 10) % 10
        return base_number + str(checksum)
    else:
        # Generate invalid check digit
        valid_checksum = sum(int(d) for d in base_number) % 10
        invalid_checksum = (valid_checksum + np.random.randint(1, 9)) % 10
        return base_number + str(invalid_checksum)

def generate_dataset(n_samples=1000, fraud_rate=0.01):
    """
    Generate dataset with billing address, IP address, region, and card number.
    Fraud is determined by invalid credit card numbers.
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

    # Generate legitimate transactions (valid card numbers)
    legitimate_data = []
    for _ in range(n_legitimate):
        # Pick a random region
        region = np.random.choice(list(regions.keys()))
        # Generate matching IP
        ip_prefix = np.random.choice(regions[region])
        ip_suffix = np.random.randint(0, 256)
        ip_address = f"{ip_prefix}.{ip_suffix}.0.0"

        # Generate valid card number
        card_number = generate_card_number(is_valid=True)

        legitimate_data.append({
            'ip_address': ip_address,
            'billing_region': region,
            'card_number': card_number,
            'is_card_valid': 1,
            'is_fraud': 0
        })

    # Generate fraudulent transactions (invalid card numbers)
    fraud_data = []
    for _ in range(n_fraud):
        # Generate random region and IP
        region = np.random.choice(list(regions.keys()))
        ip_prefix = np.random.choice(regions[region])
        ip_suffix = np.random.randint(0, 256)
        ip_address = f"{ip_prefix}.{ip_suffix}.0.0"

        # Generate invalid card number
        card_number = generate_card_number(is_valid=False)

        fraud_data.append({
            'ip_address': ip_address,
            'billing_region': region,
            'card_number': card_number,
            'is_card_valid': 0,
            'is_fraud': 1
        })

    # Combine and shuffle data
    df = pd.DataFrame(legitimate_data + fraud_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

def prepare_data(df):
    """Prepare dataset for training"""
    # Extract region from IP address
    df['ip_region'] = df['ip_address'].apply(lambda x: 'Asia' if any(prefix in x for prefix in ['223', '218', '202', '211'])
                                           else 'Europe' if any(prefix in x for prefix in ['176', '151'])
                                           else 'North America')

    # Create feature columns
    df['ip_region_mismatch'] = (df['ip_region'] != df['billing_region']).astype(int)

    # Prepare features and target
    X = df[['ip_region_mismatch', 'is_card_valid']].values
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
    dataset_df = generate_dataset(n_samples=n_samples, fraud_rate=fraud_rate)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print("Total Samples:", len(dataset_df))
    print("Fraud Distribution:")
    print(dataset_df['is_fraud'].value_counts())
    print("\nFeature Statistics:")
    print("Invalid Card Numbers:", sum(dataset_df['is_card_valid'] == 0))
    dataset_df['ip_region'] = dataset_df['ip_address'].apply(lambda x: 'Asia' if any(prefix in x for prefix in ['223', '218', '202', '211'])
                                           else 'Europe' if any(prefix in x for prefix in ['176', '151'])
                                           else 'North America')
    print("IP-Billing Region Mismatches:", sum(dataset_df['ip_region'] != dataset_df['billing_region']))

    # Prepare data
    X, y = prepare_data(dataset_df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare PyTorch tensors for neural networks
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

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
    features = ['ip_region_mismatch', 'is_card_valid']
    for feature, importance in zip(features, rf.feature_importances_):
        print(f"{feature}: {importance:.3f}")

    results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))

    # Compare all models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame(results)
    comparison_df.set_index('model', inplace=True)
    print("\nMetrics for all models:")
    print(comparison_df)

    return comparison_df, dataset_df

if __name__ == "__main__":
    # Create directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Set parameters
    n_samples = 10000  # Increased sample size
    fraud_rate = 0.01  # 1% fraud rate

    print(f"\nGenerating and evaluating dataset with {fraud_rate*100}% fraud rate")
    print(f"Total samples: {n_samples}")
    print(f"Expected fraud cases: {int(n_samples * fraud_rate)}")

    # Generate and evaluate models
    results_df, dataset_df = train_and_evaluate_all_models(n_samples=n_samples, fraud_rate=fraud_rate)

    # Save results
    results_df.to_csv(f'results/card_testing_results_rate_{fraud_rate}.csv')
    dataset_df.to_csv(f'results/card_testing_dataset_rate_{fraud_rate}.csv', index=False)

    print("\nResults have been saved to the 'results' directory.")
