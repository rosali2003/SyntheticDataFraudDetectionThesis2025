import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import ipaddress
from typing import List, Tuple

class GeolocationMismatchDataset:
    def __init__(self, num_samples: int = 10000, fraud_ratio: float = 0.1):
        """
        Initialize the dataset generator for geolocation mismatch scenarios.

        Args:
            num_samples: Total number of transactions to generate
            fraud_ratio: Ratio of fraudulent transactions (default 10%)
        """
        self.num_samples = num_samples
        self.fraud_ratio = fraud_ratio
        self.faker = Faker()

        # Define regions for generating locations
        self.regions = ['NA', 'EU', 'APAC', 'LATAM', 'MEA']
        self.high_risk_regions = ['MEA', 'APAC']  # Regions with higher fraud rates

        # Define common proxy servers and VPN locations
        self.proxy_locations = ['Netherlands', 'Romania', 'Russia', 'Hong Kong', 'Singapore']

    def generate_ip(self) -> str:
        """Generate a random IP address."""
        return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))

    def generate_device_fingerprint(self) -> dict:
        """Generate device fingerprint details."""
        browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
        os_types = ['Windows', 'MacOS', 'Linux', 'iOS', 'Android']

        return {
            'browser': random.choice(browsers),
            'os': random.choice(os_types),
            'screen_resolution': f"{random.choice([1366, 1920, 2560])}x{random.choice([768, 1080, 1440])}",
            'language': random.choice(['en-US', 'en-GB', 'es-ES', 'fr-FR', 'de-DE'])
        }

    def generate_transaction(self, is_fraud: bool) -> dict:
        """Generate a single transaction with location details."""
        billing_region = random.choice(self.regions)

        if is_fraud:
            # For fraud cases, create location mismatches
            shipping_region = random.choice([r for r in self.regions if r != billing_region])
            ip_location = random.choice(self.proxy_locations)
            is_proxy = random.choice([True, True, False])  # Higher chance of proxy usage in fraud
            device = self.generate_device_fingerprint()
            email = f"{self.faker.user_name()}{random.randint(1000,9999)}@{random.choice(['tempmail.com', 'fakemail.org', 'burner.net'])}"
        else:
            # For legitimate cases, maintain consistent locations
            shipping_region = billing_region
            ip_location = self.faker.country()
            is_proxy = random.choice([True, False, False, False])  # Lower chance of proxy usage
            device = self.generate_device_fingerprint()
            email = self.faker.email()

        return {
            'transaction_id': self.faker.uuid4(),
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            'ip_address': self.generate_ip(),
            'ip_location': ip_location,
            'billing_region': billing_region,
            'shipping_region': shipping_region,
            'email': email,
            'is_proxy': is_proxy,
            'device_fingerprint': device,
            'amount': round(random.uniform(10, 1000), 2),
            'is_fraud': is_fraud
        }

    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset."""
        fraud_samples = int(self.num_samples * self.fraud_ratio)
        normal_samples = self.num_samples - fraud_samples

        # Generate fraud and normal transactions
        fraud_transactions = [self.generate_transaction(True) for _ in range(fraud_samples)]
        normal_transactions = [self.generate_transaction(False) for _ in range(normal_samples)]

        # Combine all transactions
        all_transactions = fraud_transactions + normal_transactions
        random.shuffle(all_transactions)

        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)

        # Normalize device fingerprint columns
        device_df = pd.json_normalize(df['device_fingerprint'])
        df = pd.concat([df.drop('device_fingerprint', axis=1), device_df], axis=1)

        return df

def create_dataset(num_samples: int = 10000, fraud_ratio: float = 0.1, output_file: str = 'geolocation_mismatch_dataset.csv'):
    """
    Create and save the geolocation mismatch dataset.

    Args:
        num_samples: Number of transactions to generate
        fraud_ratio: Ratio of fraudulent transactions
        output_file: Name of the output CSV file
    """
    generator = GeolocationMismatchDataset(num_samples, fraud_ratio)
    df = generator.generate_dataset()
    df.to_csv(output_file, index=False)
    print(f"Dataset generated with {num_samples} samples ({fraud_ratio*100}% fraud)")
    print(f"Dataset saved to {output_file}")
    return df

if __name__ == "__main__":
    create_dataset()
