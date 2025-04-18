# Add imports from card_testing_6.py
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

# Define merchant category codes and their typical transaction ranges
MCC_RANGES = {
    # Retail stores
    '5411': (10, 500),     # Grocery stores
    '5912': (5, 200),      # Drug stores
    '5811': (20, 300),     # Restaurants
    '5814': (5, 100),      # Fast food

    # Services
    '4121': (10, 200),     # Taxi/Rideshare
    '7523': (5, 100),      # Parking lots
    '5541': (20, 150),     # Gas stations

    # Entertainment
    '7832': (10, 50),      # Movie theaters
    '7922': (50, 500),     # Theaters/concerts
    '7991': (20, 200),     # Tourist attractions

    # Travel
    '4111': (1, 500),      # Local transport
    '4511': (100, 2000),   # Airlines
    '7011': (100, 1000),   # Hotels

    # Professional services
    '8011': (50, 1000),    # Doctors
    '8021': (100, 2000),   # Dentists
    '8062': (500, 5000),   # Hospitals
}

# Define typical transaction type distributions for each MCC
MCC_TRANSACTION_TYPES = {
    # Retail stores - mix of all types
    '5411': {'credit': 0.4, 'debit': 0.4, 'cash': 0.2},  # Grocery stores
    '5912': {'credit': 0.3, 'debit': 0.5, 'cash': 0.2},  # Drug stores
    '5811': {'credit': 0.6, 'debit': 0.3, 'cash': 0.1},  # Restaurants
    '5814': {'credit': 0.2, 'debit': 0.5, 'cash': 0.3},  # Fast food

    # Services - mostly debit and cash
    '4121': {'credit': 0.2, 'debit': 0.5, 'cash': 0.3},  # Taxi/Rideshare
    '7523': {'credit': 0.1, 'debit': 0.4, 'cash': 0.5},  # Parking lots
    '5541': {'credit': 0.3, 'debit': 0.5, 'cash': 0.2},  # Gas stations

    # Entertainment - mostly credit
    '7832': {'credit': 0.6, 'debit': 0.3, 'cash': 0.1},  # Movie theaters
    '7922': {'credit': 0.7, 'debit': 0.2, 'cash': 0.1},  # Theaters/concerts
    '7991': {'credit': 0.6, 'debit': 0.3, 'cash': 0.1},  # Tourist attractions

    # Travel - heavily credit
    '4111': {'credit': 0.5, 'debit': 0.4, 'cash': 0.1},  # Local transport
    '4511': {'credit': 0.8, 'debit': 0.2, 'cash': 0.0},  # Airlines
    '7011': {'credit': 0.8, 'debit': 0.2, 'cash': 0.0},  # Hotels

    # Professional services - mostly credit
    '8011': {'credit': 0.7, 'debit': 0.2, 'cash': 0.1},  # Doctors
    '8021': {'credit': 0.7, 'debit': 0.2, 'cash': 0.1},  # Dentists
    '8062': {'credit': 0.8, 'debit': 0.2, 'cash': 0.0},  # Hospitals
}

# Define device types and their compatible browsers
DEVICE_BROWSER_COMPATIBILITY = {
    'iPhone': {
        'Safari': True,
        'Chrome': True,
        'Firefox': True,
        'Edge': False,
        'Internet Explorer': False,
        'Samsung Internet': False,
        'Opera': True
    },
    'Android Phone': {
        'Safari': False,
        'Chrome': True,
        'Firefox': True,
        'Edge': True,
        'Internet Explorer': False,
        'Samsung Internet': True,
        'Opera': True
    },
    'iPad': {
        'Safari': True,
        'Chrome': True,
        'Firefox': True,
        'Edge': False,
        'Internet Explorer': False,
        'Samsung Internet': False,
        'Opera': True
    },
    'Android Tablet': {
        'Safari': False,
        'Chrome': True,
        'Firefox': True,
        'Edge': True,
        'Internet Explorer': False,
        'Samsung Internet': True,
        'Opera': True
    },
    'Windows PC': {
        'Safari': False,
        'Chrome': True,
        'Firefox': True,
        'Edge': True,
        'Internet Explorer': True,
        'Samsung Internet': False,
        'Opera': True
    },
    'Mac': {
        'Safari': True,
        'Chrome': True,
        'Firefox': True,
        'Edge': True,
        'Internet Explorer': False,
        'Samsung Internet': False,
        'Opera': True
    },
    'Linux PC': {
        'Safari': False,
        'Chrome': True,
        'Firefox': True,
        'Edge': True,
        'Internet Explorer': False,
        'Samsung Internet': False,
        'Opera': True
    }
}

# Device type distribution by region
REGION_DEVICE_DISTRIBUTION = {
    'Asia': {
        'iPhone': 0.25,
        'Android Phone': 0.45,
        'iPad': 0.05,
        'Android Tablet': 0.05,
        'Windows PC': 0.15,
        'Mac': 0.03,
        'Linux PC': 0.02
    },
    'Europe': {
        'iPhone': 0.30,
        'Android Phone': 0.35,
        'iPad': 0.05,
        'Android Tablet': 0.05,
        'Windows PC': 0.18,
        'Mac': 0.05,
        'Linux PC': 0.02
    },
    'North America': {
        'iPhone': 0.45,
        'Android Phone': 0.25,
        'iPad': 0.07,
        'Android Tablet': 0.03,
        'Windows PC': 0.12,
        'Mac': 0.06,
        'Linux PC': 0.02
    }
}

# Browser distribution by region
REGION_BROWSER_DISTRIBUTION = {
    'Asia': {
        'Safari': 0.15,
        'Chrome': 0.50,
        'Firefox': 0.10,
        'Edge': 0.05,
        'Internet Explorer': 0.05,
        'Samsung Internet': 0.10,
        'Opera': 0.05
    },
    'Europe': {
        'Safari': 0.20,
        'Chrome': 0.45,
        'Firefox': 0.15,
        'Edge': 0.08,
        'Internet Explorer': 0.02,
        'Samsung Internet': 0.05,
        'Opera': 0.05
    },
    'North America': {
        'Safari': 0.35,
        'Chrome': 0.40,
        'Firefox': 0.10,
        'Edge': 0.08,
        'Internet Explorer': 0.02,
        'Samsung Internet': 0.03,
        'Opera': 0.02
    }
}

# Define locations for each region
REGION_LOCATIONS = {
    'Asia': [
        'Tokyo, Japan',
        'Shanghai, China',
        'Seoul, South Korea',
        'Singapore',
        'Mumbai, India',
        'Bangkok, Thailand'
    ],
    'Europe': [
        'London, UK',
        'Paris, France',
        'Berlin, Germany',
        'Rome, Italy',
        'Madrid, Spain',
        'Amsterdam, Netherlands'
    ],
    'North America': [
        'New York, USA',
        'Los Angeles, USA',
        'Chicago, USA',
        'Toronto, Canada',
        'Mexico City, Mexico',
        'Vancouver, Canada'
    ]
}

# Add new authentication method distributions
AUTHENTICATION_METHODS = {
    'SMS OTP': 0.35,
    '2FA App': 0.25,
    'Email OTP': 0.20,
    'Security Questions': 0.15,
    'None': 0.05  # No authentication
}

# Authentication method distribution by region
REGION_AUTH_DISTRIBUTION = {
    'Asia': {
        'SMS OTP': 0.40,
        '2FA App': 0.20,
        'Email OTP': 0.20,
        'Security Questions': 0.15,
        'None': 0.05
    },
    'Europe': {
        'SMS OTP': 0.30,
        '2FA App': 0.30,
        'Email OTP': 0.20,
        'Security Questions': 0.15,
        'None': 0.05
    },
    'North America': {
        'SMS OTP': 0.35,
        '2FA App': 0.25,
        'Email OTP': 0.20,
        'Security Questions': 0.15,
        'None': 0.05
    }
}

def generate_authentication_method(region, is_fraud=False):
    """
    Generate authentication method based on region distribution
    For fraudulent transactions, return None (no authentication)
    """
    if is_fraud:
        return 'None'

    return np.random.choice(
        list(REGION_AUTH_DISTRIBUTION[region].keys()),
        p=list(REGION_AUTH_DISTRIBUTION[region].values())
    )

def generate_device_browser(region, is_fraud=False):
    """
    Generate device and browser combination based on region distributions
    For fraudulent transactions, occasionally generate incompatible combinations
    """
    if is_fraud and np.random.random() < 0.3:  # 30% chance of incompatible combination for fraud
        device = np.random.choice(list(DEVICE_BROWSER_COMPATIBILITY.keys()))
        # Intentionally choose an incompatible browser
        compatible_browsers = [browser for browser, is_compatible in DEVICE_BROWSER_COMPATIBILITY[device].items()
                            if not is_compatible]
        if compatible_browsers:
            browser = np.random.choice(compatible_browsers)
        else:
            browser = np.random.choice(list(DEVICE_BROWSER_COMPATIBILITY[device].keys()))
    else:
        # Generate based on region distribution
        device = np.random.choice(
            list(REGION_DEVICE_DISTRIBUTION[region].keys()),
            p=list(REGION_DEVICE_DISTRIBUTION[region].values())
        )
        browser = np.random.choice(
            list(REGION_BROWSER_DISTRIBUTION[region].keys()),
            p=list(REGION_BROWSER_DISTRIBUTION[region].values())
        )

    is_compatible = DEVICE_BROWSER_COMPATIBILITY[device][browser]
    return device, browser, is_compatible

def generate_dataset(n_samples=5000, fraud_rate=0.01):
    """
    Generate dataset with location, IP address, region, card number, transaction amount,
    MCC, transaction type, device type, browser information, and authentication method.
    Fraud is determined by:
    1. Invalid credit card numbers AND
    2. Transaction amount > $200 AND
    3. MCC consistent with transaction amount being below $2000 AND
    4. Location does not match IP region AND
    5. Device type is compatible with browser AND
    6. Authentication method is None
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

    # Generate legitimate transactions
    legitimate_data = []
    for _ in range(n_legitimate):
        # Pick a random region and matching location
        region = np.random.choice(list(regions.keys()))
        location = np.random.choice(REGION_LOCATIONS[region])

        # Generate matching IP
        ip_prefix = np.random.choice(regions[region])
        ip_suffix = np.random.randint(0, 256)
        ip_address = f"{ip_prefix}.{ip_suffix}.0.0"

        # Generate device and browser information
        device, browser, is_compatible = generate_device_browser(region, is_fraud=False)

        # Generate authentication method
        auth_method = generate_authentication_method(region, is_fraud=False)

        # Generate other transaction details
        card_number = generate_card_number(is_valid=True)
        mcc = np.random.choice(list(MCC_RANGES.keys()))
        min_amount, max_amount = MCC_RANGES[mcc]
        amount = np.random.uniform(min_amount, max_amount)
        trans_type = generate_transaction_type(mcc, is_fraud=False)

        legitimate_data.append({
            'ip_address': ip_address,
            'location': location,
            'card_number': card_number,
            'transaction_amount': amount,
            'mcc': mcc,
            'transaction_type': trans_type,
            'device_type': device,
            'browser': browser,
            'authentication_method': auth_method,
            'is_device_browser_compatible': int(is_compatible),
            'is_card_valid': 1,
            'is_fraud': 0
        })

    # Generate fraudulent transactions
    fraud_data = []
    for _ in range(n_fraud):
        # Generate mismatched location and IP region
        ip_region = np.random.choice(list(regions.keys()))
        location_region = np.random.choice([r for r in regions.keys() if r != ip_region])
        location = np.random.choice(REGION_LOCATIONS[location_region])

        # Generate IP from different region
        ip_prefix = np.random.choice(regions[ip_region])
        ip_suffix = np.random.randint(0, 256)
        ip_address = f"{ip_prefix}.{ip_suffix}.0.0"

        # Generate compatible device and browser combination (for fraud)
        device = np.random.choice(list(DEVICE_BROWSER_COMPATIBILITY.keys()))
        compatible_browsers = [browser for browser, is_compatible in DEVICE_BROWSER_COMPATIBILITY[device].items()
                            if is_compatible]
        browser = np.random.choice(compatible_browsers)
        is_compatible = True  # We ensure compatibility for fraud cases

        # Generate authentication method (None for fraud)
        auth_method = generate_authentication_method(ip_region, is_fraud=True)

        # Generate other transaction details
        card_number = generate_card_number(is_valid=False)
        low_amount_mccs = [mcc for mcc, (_, max_amt) in MCC_RANGES.items() if max_amt < 2000]
        mcc = np.random.choice(low_amount_mccs)
        amount = np.random.uniform(200, 1999)  # Amount > $200 but within MCC typical range
        trans_type = generate_transaction_type(mcc, is_fraud=True)

        fraud_data.append({
            'ip_address': ip_address,
            'location': location,
            'card_number': card_number,
            'transaction_amount': amount,
            'mcc': mcc,
            'transaction_type': trans_type,
            'device_type': device,
            'browser': browser,
            'authentication_method': auth_method,
            'is_device_browser_compatible': int(is_compatible),
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

    # Extract region from location
    df['location_region'] = df['location'].apply(get_region_from_location)

    # Create feature columns
    df['location_ip_mismatch'] = (df['ip_region'] != df['location_region']).astype(int)
    df['high_amount'] = (df['transaction_amount'] > 200).astype(int)

    # Add MCC suspicion feature
    df['mcc_amount_mismatch'] = df.apply(
        lambda row: is_amount_suspicious_for_mcc(row['transaction_amount'], row['mcc']),
        axis=1
    ).astype(int)

    # Convert categorical variables to one-hot encoding
    transaction_type_dummies = pd.get_dummies(df['transaction_type'], prefix='trans_type')
    device_type_dummies = pd.get_dummies(df['device_type'], prefix='device')
    browser_dummies = pd.get_dummies(df['browser'], prefix='browser')
    auth_method_dummies = pd.get_dummies(df['authentication_method'], prefix='auth')

    # Select numerical features that need scaling
    numerical_features = ['transaction_amount']

    # Create a copy of numerical features for scaling
    X_numerical = df[numerical_features].values

    # Create binary features
    binary_features = [
        'location_ip_mismatch',
        'is_card_valid',
        'mcc_amount_mismatch',
        'is_device_browser_compatible'
    ]
    X_binary = df[binary_features].values

    # Combine all features
    X = np.hstack([
        X_binary,
        X_numerical,
        transaction_type_dummies.values,
        device_type_dummies.values,
        browser_dummies.values,
        auth_method_dummies.values
    ])

    y = df['is_fraud'].values

    return X, y

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

def is_amount_suspicious_for_mcc(amount, mcc):
    """
    Check if transaction amount is suspiciously high for the given MCC
    """
    if mcc in MCC_RANGES:
        _, max_typical = MCC_RANGES[mcc]
        return amount > max_typical * 2  # Consider suspicious if more than 2x typical max
    return False

def get_region_from_location(location):
    """
    Get the region based on the location
    """
    for region, locations in REGION_LOCATIONS.items():
        if any(loc in location for loc in locations):
            return region
    return None

def generate_transaction_type(mcc, is_fraud=False):
    """
    Generate transaction type based on MCC typical distributions
    For fraudulent transactions, prefer credit cards
    """
    if is_fraud:
        # Fraudulent transactions are more likely to be credit
        return np.random.choice(['credit', 'debit'], p=[0.9, 0.1])

    if mcc in MCC_TRANSACTION_TYPES:
        dist = MCC_TRANSACTION_TYPES[mcc]
        return np.random.choice(
            ['credit', 'debit', 'cash'],
            p=[dist['credit'], dist['debit'], dist['cash']]
        )
    return np.random.choice(['credit', 'debit', 'cash'])

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
    print("High Amount Transactions (>$200):", sum(dataset_df['transaction_amount'] > 200))
    print("Authentication Method Distribution:")
    print(dataset_df['authentication_method'].value_counts())
    print("\nDevice-Browser Statistics:")
    print("Device Type Distribution:")
    print(dataset_df['device_type'].value_counts())
    print("Browser Distribution:")
    print(dataset_df['browser'].value_counts())
    print("Device-Browser Compatibility:")
    print("Compatible:", sum(dataset_df['is_device_browser_compatible'] == 1))
    print("Incompatible:", sum(dataset_df['is_device_browser_compatible'] == 0))

    # Calculate location-IP mismatches
    dataset_df['ip_region'] = dataset_df['ip_address'].apply(lambda x: 'Asia' if any(prefix in x for prefix in ['223', '218', '202', '211'])
                                           else 'Europe' if any(prefix in x for prefix in ['176', '151'])
                                           else 'North America')
    dataset_df['location_region'] = dataset_df['location'].apply(get_region_from_location)
    print("Location-IP Mismatches:", sum(dataset_df['ip_region'] != dataset_df['location_region']))

    # Define feature groups before prepare_data
    binary_features = [
        'location_ip_mismatch',
        'is_card_valid',
        'mcc_amount_mismatch',
        'is_device_browser_compatible'
    ]
    numerical_features = ['transaction_amount']

    # Get one-hot encoded column names
    transaction_type_columns = [f'trans_type_{t}' for t in dataset_df['transaction_type'].unique()]
    device_type_columns = [f'device_{t}' for t in dataset_df['device_type'].unique()]
    browser_columns = [f'browser_{t}' for t in dataset_df['browser'].unique()]
    auth_method_columns = [f'auth_{t}' for t in dataset_df['authentication_method'].unique()]

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

    # Train and evaluate models
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log = log_reg.predict(X_test_scaled)
    results.append(evaluate_model(y_test, y_pred_log, "Logistic Regression"))

    # 2. Linear Regression
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

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = single_layer(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    single_layer.eval()
    with torch.no_grad():
        y_pred_single = (single_layer(X_test_tensor) >= 0.5).float().numpy()
    results.append(evaluate_model(y_test, y_pred_single, "Single Layer Neural Network"))

    # 4. Two Layer Neural Network
    print("\nTraining Two Layer Neural Network...")
    two_layer = TwoLayerNet(input_size=X.shape[1])
    optimizer = torch.optim.Adam(two_layer.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = two_layer(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

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
    all_features = (
        binary_features +  # Binary features
        numerical_features +  # Numerical features
        transaction_type_columns +  # Transaction type features
        device_type_columns +  # Device type features
        browser_columns +  # Browser features
        auth_method_columns  # Authentication method features
    )

    for feature, importance in zip(all_features, rf.feature_importances_):
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
    n_samples = 10000  # Fixed sample size
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
