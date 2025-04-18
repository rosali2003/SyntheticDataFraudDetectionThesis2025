import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

def create_dataset(n_samples=100, fraud_rate=0.01):
    # Detailed regions and their cities
    regions_data = {
        'North America': {
            'cities': ['New York, USA', 'Toronto, Canada', 'Los Angeles, USA', 'Chicago, USA',
                      'Vancouver, Canada', 'Mexico City, Mexico', 'Montreal, Canada'],
            'ip_ranges': ['24', '50', '76', '99'],  # First octet of IP ranges
            'typical_amount_range': (10, 1000),
            'timezone_offset': -5  # EST
        },
        'Europe': {
            'cities': ['London, UK', 'Paris, France', 'Berlin, Germany', 'Madrid, Spain',
                      'Rome, Italy', 'Amsterdam, Netherlands', 'Vienna, Austria'],
            'ip_ranges': ['128', '151', '176', '188'],
            'typical_amount_range': (10, 850),
            'timezone_offset': 1  # CET
        },
        'Asia': {
            'cities': ['Tokyo, Japan', 'Seoul, South Korea', 'Beijing, China', 'Singapore',
                      'Mumbai, India', 'Bangkok, Thailand', 'Hong Kong'],
            'ip_ranges': ['202', '211', '218', '223'],
            'typical_amount_range': (10, 750),
            'timezone_offset': 8  # CST
        }
    }

    data = {
        'ip_address': [],
        'billing_address': [],
        'region': [],
        'transaction_amount': [],
        'transaction_time': [],
        'transaction_count_1h': [],
        'is_fraud': []
    }

    # Calculate exact number of fraud cases
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legitimate = n_samples - n_fraud

    street_types = ['Street', 'Avenue', 'Boulevard', 'Road', 'Lane']
    street_names = ['Maple', 'Oak', 'Pine', 'Cedar', 'Elm', 'Main', 'Park', 'Lake']

    def generate_transaction_time(region, is_fraud=False):
        now = datetime.now()
        if is_fraud:
            # Fraudulent transactions are more likely (but not always) during odd hours
            if random.random() < 0.7:  # 70% chance of odd hours
                hour = random.choice([0, 1, 2, 3, 4, 23])
            else:
                hour = random.randint(0, 23)  # 30% chance of any hour
        else:
            # Legitimate transactions more likely (but not always) during business hours
            if random.random() < 0.8:  # 80% chance of business hours
                hour = random.randint(9, 17)
            else:
                hour = random.randint(0, 23)  # 20% chance of any hour

        offset = regions_data[region]['timezone_offset']
        time = now.replace(hour=hour, minute=random.randint(0, 59))
        return time + timedelta(hours=offset)

    def generate_transaction_amount(region, is_fraud=False):
        base_range = regions_data[region]['typical_amount_range']
        if is_fraud:
            # Fraudulent transactions more likely (but not always) to be unusual amounts
            if random.random() < 0.7:  # 70% chance of unusual amount
                if random.random() < 0.5:
                    return round(random.uniform(0.1, 5), 2)  # Small amount
                else:
                    return round(random.uniform(base_range[1], base_range[1] * 3), 2)  # Large amount
            else:
                return round(random.uniform(*base_range), 2)  # 30% chance of normal amount
        return round(random.uniform(*base_range), 2)

    def generate_transaction_count():
        # Use Poisson distribution for more realistic transaction counts
        if random.random() < 0.7:  # 70% chance of normal velocity
            return np.random.poisson(2)  # Average of 2 transactions
        else:
            return np.random.poisson(6)  # Average of 6 transactions

    # Generate legitimate cases
    for _ in range(n_legitimate):
        region = random.choice(list(regions_data.keys()))
        ip_first_octet = random.choice(regions_data[region]['ip_ranges'])
        ip_address = f"{ip_first_octet}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

        city = random.choice(regions_data[region]['cities'])
        street_number = random.randint(1, 999)
        billing_address = f"{street_number} {random.choice(street_names)} {random.choice(street_types)}, {city}"

        amount = generate_transaction_amount(region, is_fraud=False)
        trans_time = generate_transaction_time(region, is_fraud=False)
        trans_count = generate_transaction_count()

        data['ip_address'].append(ip_address)
        data['billing_address'].append(billing_address)
        data['region'].append(region)
        data['transaction_amount'].append(amount)
        data['transaction_time'].append(trans_time)
        data['transaction_count_1h'].append(trans_count)
        data['is_fraud'].append(0)

    # Generate fraud cases
    for _ in range(n_fraud):
        # Randomly choose fraud pattern
        fraud_pattern = random.choice(['region_mismatch', 'amount', 'velocity', 'time'])

        if fraud_pattern == 'region_mismatch' and random.random() < 0.8:  # 80% chance of mismatch for this pattern
            ip_region = random.choice(list(regions_data.keys()))
            billing_region = random.choice([r for r in regions_data.keys() if r != ip_region])
            ip_first_octet = random.choice(regions_data[ip_region]['ip_ranges'])
            city = random.choice(regions_data[billing_region]['cities'])
        else:
            ip_region = random.choice(list(regions_data.keys()))
            billing_region = ip_region
            ip_first_octet = random.choice(regions_data[ip_region]['ip_ranges'])
            city = random.choice(regions_data[ip_region]['cities'])

        ip_address = f"{ip_first_octet}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
        street_number = random.randint(1, 999)
        billing_address = f"{street_number} {random.choice(street_names)} {random.choice(street_types)}, {city}"

        amount = generate_transaction_amount(ip_region, is_fraud=True)
        trans_time = generate_transaction_time(ip_region, is_fraud=True)
        trans_count = generate_transaction_count()

        data['ip_address'].append(ip_address)
        data['billing_address'].append(billing_address)
        data['region'].append(ip_region)
        data['transaction_amount'].append(amount)
        data['transaction_time'].append(trans_time)
        data['transaction_count_1h'].append(trans_count)
        data['is_fraud'].append(1)

    # Create DataFrame and shuffle rows
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)

    # Add derived features (now based on actual patterns rather than direct indicators)
    df['transaction_hour'] = df['transaction_time'].apply(lambda x: x.hour)
    df['is_business_hour'] = df['transaction_hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
    df['is_high_risk_amount'] = df.apply(
        lambda row: 1 if (row['transaction_amount'] < 5 or
                         row['transaction_amount'] > regions_data[row['region']]['typical_amount_range'][1])
        else 0, axis=1
    )
    df['is_suspicious_time'] = df['transaction_hour'].apply(
        lambda x: 1 if x in [0, 1, 2, 3, 4, 23] else 0
    )
    df['is_velocity_alert'] = df['transaction_count_1h'].apply(
        lambda x: 1 if x > 4 else 0
    )

    # Save to CSV
    output_path = 'ip_billing_dataset.csv'
    df.to_csv(output_path, index=False)

    # Print summary
    print("\nDataset Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Fraud cases: {df['is_fraud'].sum()}")
    print(f"Legitimate cases: {len(df) - df['is_fraud'].sum()}")
    print(f"Fraud rate: {df['is_fraud'].sum() / len(df):.2%}")
    print("\nFeature Distribution for Fraud Cases:")
    print("High Risk Amount:", df[df['is_fraud'] == 1]['is_high_risk_amount'].mean())
    print("Suspicious Time:", df[df['is_fraud'] == 1]['is_suspicious_time'].mean())
    print("Velocity Alert:", df[df['is_fraud'] == 1]['is_velocity_alert'].mean())
    print("\nFeature Distribution for Legitimate Cases:")
    print("High Risk Amount:", df[df['is_fraud'] == 0]['is_high_risk_amount'].mean())
    print("Suspicious Time:", df[df['is_fraud'] == 0]['is_suspicious_time'].mean())
    print("Velocity Alert:", df[df['is_fraud'] == 0]['is_velocity_alert'].mean())
    print("\nFirst few rows:")
    print(df.head())
    print("\nRegion Distribution:")
    print(df['region'].value_counts())

    return df

# Generate dataset
df = create_dataset(n_samples=10000, fraud_rate=0.01)
