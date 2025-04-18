import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import os

# Constants from card_testing_8
MCC_RANGES = {
    'retail': (10, 100),
    'travel': (100, 1000),
    'entertainment': (20, 200),
    'dining': (20, 150),
    'services': (50, 500)
}

MCC_TRANSACTION_TYPES = {
    'retail': ['in_store', 'online'],
    'travel': ['online', 'in_store'],
    'entertainment': ['online', 'in_store'],
    'dining': ['in_store'],
    'services': ['online', 'in_store']
}

DEVICE_BROWSER_COMPATIBILITY = {
    'mobile': ['mobile_chrome', 'mobile_safari', 'mobile_firefox'],
    'desktop': ['chrome', 'firefox', 'safari', 'edge'],
    'tablet': ['mobile_chrome', 'mobile_safari', 'mobile_firefox', 'chrome']
}

REGION_DEVICE_DISTRIBUTION = {
    'Asia': {'mobile': 0.6, 'desktop': 0.3, 'tablet': 0.1},
    'Europe': {'desktop': 0.5, 'mobile': 0.4, 'tablet': 0.1},
    'North_America': {'desktop': 0.4, 'mobile': 0.5, 'tablet': 0.1}
}

REGION_BROWSER_DISTRIBUTION = {
    'Asia': {
        'mobile': {'mobile_chrome': 0.7, 'mobile_safari': 0.2, 'mobile_firefox': 0.1},
        'desktop': {'chrome': 0.6, 'firefox': 0.2, 'safari': 0.1, 'edge': 0.1},
        'tablet': {'mobile_chrome': 0.6, 'mobile_safari': 0.3, 'mobile_firefox': 0.1}
    },
    'Europe': {
        'mobile': {'mobile_chrome': 0.6, 'mobile_safari': 0.3, 'mobile_firefox': 0.1},
        'desktop': {'chrome': 0.5, 'firefox': 0.3, 'safari': 0.1, 'edge': 0.1},
        'tablet': {'mobile_chrome': 0.5, 'mobile_safari': 0.4, 'mobile_firefox': 0.1}
    },
    'North_America': {
        'mobile': {'mobile_chrome': 0.5, 'mobile_safari': 0.4, 'mobile_firefox': 0.1},
        'desktop': {'chrome': 0.4, 'firefox': 0.3, 'safari': 0.2, 'edge': 0.1},
        'tablet': {'mobile_chrome': 0.4, 'mobile_safari': 0.5, 'mobile_firefox': 0.1}
    }
}

REGION_LOCATIONS = {
    'Asia': ['Tokyo', 'Shanghai', 'Seoul', 'Singapore', 'Mumbai', 'Bangkok'],
    'Europe': ['London', 'Paris', 'Berlin', 'Rome', 'Madrid', 'Amsterdam'],
    'North_America': ['New York', 'Los Angeles', 'Chicago', 'Toronto', 'Mexico City', 'Vancouver']
}

AUTHENTICATION_METHODS = {
    'Asia': {
        '2FA': 0.4,
        'biometric': 0.3,
        'password': 0.2,
        'token': 0.1
    },
    'Europe': {
        '2FA': 0.3,
        'biometric': 0.4,
        'password': 0.2,
        'token': 0.1
    },
    'North_America': {
        '2FA': 0.3,
        'biometric': 0.3,
        'password': 0.3,
        'token': 0.1
    }
}

def luhn_checksum(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = 0
    checksum += sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10

def generate_card_number(is_valid=True):
    # Generate first 15 digits randomly
    card_number = [np.random.randint(0, 10) for _ in range(15)]

    if is_valid:
        # Calculate check digit to make it valid
        partial_sum = 0
        for i, digit in enumerate(card_number[::-1]):
            if i % 2 == 1:
                doubled = digit * 2
                partial_sum += doubled // 10 + doubled % 10
            else:
                partial_sum += digit
        check_digit = (10 - (partial_sum % 10)) % 10
    else:
        # Generate invalid check digit
        valid_check_digit = (10 - (sum(card_number) % 10)) % 10
        check_digit = (valid_check_digit + np.random.randint(1, 10)) % 10

    card_number.append(check_digit)
    return int(''.join(map(str, card_number)))

def is_amount_suspicious_for_mcc(mcc, amount):
    mcc_range = MCC_RANGES[mcc]
    return amount > 2000  # Modified: Amount threshold increased to 2000

def get_region_from_location(location):
    for region, cities in REGION_LOCATIONS.items():
        if location in cities:
            return region
    return None

def generate_authentication_method(region, is_fraud=False):
    if is_fraud:
        return None
    else:
        methods = list(AUTHENTICATION_METHODS[region].keys())
        probabilities = list(AUTHENTICATION_METHODS[region].values())
        return np.random.choice(methods, p=probabilities)

def generate_device_browser(region, is_fraud=False):
    # Select device type based on region distribution
    devices = list(REGION_DEVICE_DISTRIBUTION[region].keys())
    device_probs = list(REGION_DEVICE_DISTRIBUTION[region].values())
    device = np.random.choice(devices, p=device_probs)

    # Select browser based on region and device distribution
    browsers = list(REGION_BROWSER_DISTRIBUTION[region][device].keys())
    browser_probs = list(REGION_BROWSER_DISTRIBUTION[region][device].values())
    browser = np.random.choice(browsers, p=browser_probs)

    # For fraud transactions, ensure device and browser are compatible
    is_compatible = browser in DEVICE_BROWSER_COMPATIBILITY[device]

    return device, browser, is_compatible

def generate_transaction_type(mcc):
    return np.random.choice(MCC_TRANSACTION_TYPES[mcc])

def calculate_transactions_past_15min(df, current_timestamp):
    """Calculate number of transactions in past 15 minutes"""
    fifteen_min_ago = current_timestamp - timedelta(minutes=15)
    return len(df[df['timestamp'].between(fifteen_min_ago, current_timestamp)])

def generate_dataset(n_samples, fraud_rate=0.05):
    n_fraud = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_fraud

    data = []
    regions = list(REGION_LOCATIONS.keys())
    mccs = list(MCC_RANGES.keys())

    # Generate timestamps over a 30-day period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Generate legitimate transactions
    for _ in range(n_legitimate):
        region = np.random.choice(regions)
        mcc = np.random.choice(mccs)
        mcc_range = MCC_RANGES[mcc]
        amount = np.random.uniform(mcc_range[0], mcc_range[1])
        location = np.random.choice(REGION_LOCATIONS[region])
        device, browser, is_compatible = generate_device_browser(region, is_fraud=False)
        timestamp = start_date + (end_date - start_date) * np.random.random()

        data.append({
            'timestamp': timestamp,
            'card_number': generate_card_number(is_valid=True),
            'transaction_amount': amount,
            'mcc': mcc,
            'transaction_type': generate_transaction_type(mcc),
            'ip_region': region,
            'location': location,
            'device_type': device,
            'browser': browser,
            'authentication_method': generate_authentication_method(region, is_fraud=False),
            'is_fraud': False
        })

    # Generate fraudulent transactions with clustered timestamps
    fraud_timestamps = []
    for _ in range(n_fraud):
        # Generate clustered timestamps for fraud
        if not fraud_timestamps:
            base_timestamp = start_date + (end_date - start_date) * np.random.random()
            # Generate 15 timestamps within a 15-minute window to ensure high transaction frequency
            fraud_timestamps = [base_timestamp + timedelta(minutes=i) for i in range(15)]

        timestamp = fraud_timestamps.pop()
        region = np.random.choice(regions)
        mcc = np.random.choice(mccs)
        # Ensure amount is between 200 and 2000 for fraud
        amount = np.random.uniform(201, 1999)
        # Ensure location is from a different region
        location = np.random.choice([loc for reg in REGION_LOCATIONS.values()
                                   for loc in reg if get_region_from_location(loc) != region])
        # Generate compatible device-browser pair for fraud
        device, browser, is_compatible = generate_device_browser(region, is_fraud=True)
        while not is_compatible:  # Ensure we get a compatible pair
            device, browser, is_compatible = generate_device_browser(region, is_fraud=True)

        transaction = {
            'timestamp': timestamp,
            'card_number': generate_card_number(is_valid=False),  # Always invalid for fraud
            'transaction_amount': amount,
            'mcc': mcc,
            'transaction_type': generate_transaction_type(mcc),
            'ip_region': region,
            'location': location,
            'device_type': device,
            'browser': browser,
            'authentication_method': None,  # Always None for fraud
            'is_fraud': True
        }
        data.append(transaction)

    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate time since last transaction and transactions in past 15 minutes
    df['time_since_last'] = df['timestamp'].diff().dt.total_seconds()
    df.loc[0, 'time_since_last'] = 0  # First transaction

    # Calculate transactions in past 15 minutes for each transaction
    df['transactions_past_15min'] = df.apply(
        lambda row: calculate_transactions_past_15min(
            df[df.index < row.name],  # Only look at previous transactions
            row['timestamp']
        ),
        axis=1
    )

    # Generate additional features
    df['is_card_valid'] = df['card_number'].apply(lambda x: luhn_checksum(x) == 0)
    df['mcc_amount_mismatch'] = df.apply(lambda row:
        is_amount_suspicious_for_mcc(row['mcc'], row['transaction_amount']), axis=1)
    df['location_ip_mismatch'] = df.apply(lambda row:
        get_region_from_location(row['location']) != row['ip_region'], axis=1)
    df['device_browser_compatible'] = df.apply(lambda row:
        row['browser'] in DEVICE_BROWSER_COMPATIBILITY[row['device_type']], axis=1)

    # Update fraud labels based on all conditions
    df['is_fraud'] = (
        (~df['is_card_valid']) &  # Invalid card number
        (df['transaction_amount'] > 200) &  # Amount > $200
        (df['transaction_amount'] < 2000) &  # Amount < $2000
        (df['location_ip_mismatch']) &  # Location doesn't match IP region
        (df['device_browser_compatible']) &  # Device is compatible with browser
        (df['authentication_method'].isna()) &  # Authentication method is null
        (df['transactions_past_15min'] > 10)  # More than 10 transactions in past 15 minutes
    ).astype(int)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total Transactions: {len(df)}")
    print(f"Fraudulent Transactions: {df['is_fraud'].sum()}")
    print(f"Invalid Card Numbers: {(~df['is_card_valid']).sum()}")
    print(f"High Amount Transactions (>200): {(df['transaction_amount'] > 200).sum()}")
    print(f"Very High Amount Transactions (>2000): {(df['transaction_amount'] > 2000).sum()}")
    print(f"Location-IP Mismatches: {df['location_ip_mismatch'].sum()}")
    print(f"Device-Browser Compatible Cases: {df['device_browser_compatible'].sum()}")
    print(f"Null Authentication Methods: {df['authentication_method'].isna().sum()}")
    print(f"Transactions with >10 in past 15min: {(df['transactions_past_15min'] > 10).sum()}")

    return df

class SingleLayerNet(nn.Module):
    def __init__(self, input_size):
        super(SingleLayerNet, self).__init__()
        self.layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

class TwoLayerNet(nn.Module):
    def __init__(self, input_size):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

def prepare_data(df):
    # Select numerical features for scaling
    numerical_features = ['transaction_amount', 'time_since_last', 'transactions_past_15min']  # Added transactions_past_15min
    X_numerical = df[numerical_features].copy()

    # Binary features
    binary_features = ['is_card_valid', 'mcc_amount_mismatch', 'location_ip_mismatch',
                      'device_browser_compatible']
    X_binary = df[binary_features]

    # Categorical features for one-hot encoding
    transaction_type_dummies = pd.get_dummies(df['transaction_type'], prefix='transaction_type')
    device_type_dummies = pd.get_dummies(df['device_type'], prefix='device_type')
    browser_dummies = pd.get_dummies(df['browser'], prefix='browser')
    auth_method_dummies = pd.get_dummies(df['authentication_method'], prefix='auth_method')

    # Scale numerical features
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)

    # Combine all features
    X = np.hstack([
        X_binary,
        X_numerical_scaled,
        transaction_type_dummies,
        device_type_dummies,
        browser_dummies,
        auth_method_dummies
    ])

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

def train_and_evaluate_all_models(df, results_dir):
    # Prepare data and get feature names
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining and Testing Split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("=" * 70)

    # Get all feature names before model training
    binary_features = ['is_card_valid', 'mcc_amount_mismatch', 'location_ip_mismatch',
                      'device_browser_compatible']
    numerical_features = ['transaction_amount', 'time_since_last', 'transactions_past_15min']  # Added transactions_past_15min

    # Get one-hot encoded feature names
    transaction_type_dummies = pd.get_dummies(df['transaction_type'], prefix='transaction_type')
    device_type_dummies = pd.get_dummies(df['device_type'], prefix='device_type')
    browser_dummies = pd.get_dummies(df['browser'], prefix='browser')
    auth_method_dummies = pd.get_dummies(df['authentication_method'], prefix='auth_method')

    # Combine all feature names in the same order as in prepare_data
    all_features = (binary_features +
                   numerical_features +
                   list(transaction_type_dummies.columns) +
                   list(device_type_dummies.columns) +
                   list(browser_dummies.columns) +
                   list(auth_method_dummies.columns))

    results = []

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_lr, 'Logistic Regression'))
    print("=" * 70)

    # Linear Regression
    print("\nTraining Linear Regression...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = (lin_reg.predict(X_test) > 0.5).astype(int)
    results.append(evaluate_model(y_test, y_pred_lin, 'Linear Regression'))
    print("=" * 70)

    # Convert to PyTorch tensors for neural networks
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)

    # Single Layer Neural Network
    print("\nTraining Single Layer Neural Network...")
    single_layer_net = SingleLayerNet(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(single_layer_net.parameters())

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = single_layer_net(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred_snn = (single_layer_net(X_test_tensor) > 0.5).numpy().astype(int).reshape(-1)
    results.append(evaluate_model(y_test, y_pred_snn, 'Single Layer NN'))
    print("=" * 70)

    # Two Layer Neural Network
    print("\nTraining Two Layer Neural Network...")
    two_layer_net = TwoLayerNet(X_train.shape[1])
    optimizer = optim.Adam(two_layer_net.parameters())

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = two_layer_net(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred_tnn = (two_layer_net(X_test_tensor) > 0.5).numpy().astype(int).reshape(-1)
    results.append(evaluate_model(y_test, y_pred_tnn, 'Two Layer NN'))
    print("=" * 70)

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_rf, 'Random Forest'))
    print("=" * 70)

    # Print feature importance for Random Forest
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
    print("=" * 70)

    return results

# Main execution
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"\nGenerating dataset with {0.01*100}% fraud rate")
    df = generate_dataset(10000, 0.01)

    print(f"\nTraining and evaluating models for {0.01*100}% fraud rate")
    results = train_and_evaluate_all_models(df, results_dir)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total Transactions: {len(df)}")
    print(f"Fraudulent Transactions: {df['is_fraud'].sum()}")
    print(f"Invalid Card Numbers: {(~df['is_card_valid']).sum()}")
    print(f"High Amount Transactions (>200): {(df['transaction_amount'] > 200).sum()}")
    print(f"Very High Amount Transactions (>2000): {(df['transaction_amount'] > 2000).sum()}")
    print(f"Location-IP Mismatches: {df['location_ip_mismatch'].sum()}")
    print(f"Device-Browser Compatible Cases: {df['device_browser_compatible'].sum()}")
    print(f"Null Authentication Methods: {df['authentication_method'].isna().sum()}")
    print(f"Transactions with >10 in past 15min: {(df['transactions_past_15min'] > 10).sum()}")
    print("=" * 70)
