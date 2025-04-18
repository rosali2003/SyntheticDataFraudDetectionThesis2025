import pandas as pd
import random

def create_dataset(n_samples=1000, fraud_rate=0.01):
    # Detailed regions and their cities
    regions_data = {
        'North America': {
            'cities': ['New York, USA', 'Toronto, Canada', 'Los Angeles, USA', 'Chicago, USA',
                      'Vancouver, Canada', 'Mexico City, Mexico', 'Montreal, Canada'],
            'ip_ranges': ['24', '50', '76', '99']  # First octet of IP ranges
        },
        'Europe': {
            'cities': ['London, UK', 'Paris, France', 'Berlin, Germany', 'Madrid, Spain',
                      'Rome, Italy', 'Amsterdam, Netherlands', 'Vienna, Austria'],
            'ip_ranges': ['128', '151', '176', '188']
        },
        'Asia': {
            'cities': ['Tokyo, Japan', 'Seoul, South Korea', 'Beijing, China', 'Singapore',
                      'Mumbai, India', 'Bangkok, Thailand', 'Hong Kong'],
            'ip_ranges': ['202', '211', '218', '223']
        }
    }

    data = {
        'ip_address': [],
        'billing_address': [],
        'region': [],
        'is_fraud': []
    }

    # Calculate exact number of fraud cases (1%)
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legitimate = n_samples - n_fraud

    street_types = ['Street', 'Avenue', 'Boulevard', 'Road', 'Lane']
    street_names = ['Maple', 'Oak', 'Pine', 'Cedar', 'Elm', 'Main', 'Park', 'Lake']

    # Generate legitimate cases
    for _ in range(n_legitimate):
        region = random.choice(list(regions_data.keys()))
        ip_first_octet = random.choice(regions_data[region]['ip_ranges'])
        ip_address = f"{ip_first_octet}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

        city = random.choice(regions_data[region]['cities'])
        street_number = random.randint(1, 999)
        billing_address = f"{street_number} {random.choice(street_names)} {random.choice(street_types)}, {city}"

        data['ip_address'].append(ip_address)
        data['billing_address'].append(billing_address)
        data['region'].append(region)
        data['is_fraud'].append(0)

    # Generate fraud cases
    for _ in range(n_fraud):
        ip_region = random.choice(list(regions_data.keys()))
        billing_region = random.choice([r for r in regions_data.keys() if r != ip_region])

        ip_first_octet = random.choice(regions_data[ip_region]['ip_ranges'])
        ip_address = f"{ip_first_octet}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

        city = random.choice(regions_data[billing_region]['cities'])
        street_number = random.randint(1, 999)
        billing_address = f"{street_number} {random.choice(street_names)} {random.choice(street_types)}, {city}"

        data['ip_address'].append(ip_address)
        data['billing_address'].append(billing_address)
        data['region'].append(ip_region)
        data['is_fraud'].append(1)

    # Create DataFrame and shuffle rows
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)

    # Save to CSV
    output_path = 'ip_billing_dataset.csv'
    df.to_csv(output_path, index=False)

    # Print summary
    print("\nDataset Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Fraud cases: {df['is_fraud'].sum()}")
    print(f"Legitimate cases: {len(df) - df['is_fraud'].sum()}")
    print(f"Fraud rate: {df['is_fraud'].sum() / len(df):.2%}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nRegion Distribution:")
    print(df['region'].value_counts())

    return df

# Generate dataset
df = create_dataset(n_samples=100, fraud_rate=0.01)
