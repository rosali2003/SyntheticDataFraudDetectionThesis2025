import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import uuid

class PaymentAnomalyDataset:
    def __init__(self, num_samples: int = 10000, fraud_ratio: float = 0.1):
        """
        Initialize the dataset generator for payment anomaly scenarios.

        Args:
            num_samples: Total number of transactions to generate
            fraud_ratio: Ratio of fraudulent transactions (default 10%)
        """
        self.num_samples = num_samples
        self.fraud_ratio = fraud_ratio
        self.faker = Faker()

        # Payment method types
        self.payment_methods = ['credit_card', 'debit_card', 'crypto', 'bank_transfer']
        self.crypto_currencies = ['BTC', 'ETH', 'USDT', 'XRP']

        # Time patterns
        self.normal_hours = list(range(6, 23))  # 6 AM to 11 PM
        self.unusual_hours = list(range(0, 6))  # 12 AM to 5 AM

        # Initialize user and card pools for generating realistic patterns
        self.user_pool = self._create_user_pool(int(num_samples * 0.1))  # 10% of total samples
        self.card_pool = self._create_card_pool(int(num_samples * 0.2))  # 20% of total samples

    def _create_user_pool(self, num_users: int) -> List[Dict]:
        """Create a pool of users for consistent data generation."""
        return [{
            'user_id': str(uuid.uuid4()),
            'email': self.faker.email(),
            'account_age_days': random.randint(1, 365),
            'typical_payment_methods': random.sample(self.payment_methods, random.randint(1, 3))
        } for _ in range(num_users)]

    def _create_card_pool(self, num_cards: int) -> List[Dict]:
        """Create a pool of payment cards for consistent data generation."""
        return [{
            'card_id': str(uuid.uuid4()),
            'card_type': random.choice(['credit_card', 'debit_card']),
            'issuer': random.choice(['Visa', 'Mastercard', 'Amex']),
            'last_digits': str(random.randint(1000, 9999)),
            'is_stolen': random.random() < 0.1  # 10% of cards marked as stolen
        } for _ in range(num_cards)]

    def generate_payment_method(self, is_fraud: bool, user: Dict) -> Dict:
        """Generate payment method details."""
        if is_fraud:
            if random.random() < 0.3:  # 30% chance of crypto for fraud
                return {
                    'method': 'crypto',
                    'currency': random.choice(self.crypto_currencies),
                    'wallet_address': self.faker.sha256()
                }
            else:
                card = random.choice(self.card_pool)
                return {
                    'method': card['card_type'],
                    'card_id': card['card_id'],
                    'issuer': card['issuer'],
                    'last_digits': card['last_digits']
                }
        else:
            payment_method = random.choice(user['typical_payment_methods'])
            if payment_method == 'crypto':
                return {
                    'method': 'crypto',
                    'currency': random.choice(self.crypto_currencies),
                    'wallet_address': self.faker.sha256()
                }
            else:
                card = random.choice([c for c in self.card_pool if not c['is_stolen']])
                return {
                    'method': card['card_type'],
                    'card_id': card['card_id'],
                    'issuer': card['issuer'],
                    'last_digits': card['last_digits']
                }

    def generate_transaction(self, is_fraud: bool) -> Dict:
        """Generate a single transaction with payment anomaly patterns."""
        user = random.choice(self.user_pool)

        if is_fraud:
            # Fraud patterns
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.choice(self.unusual_hours)
            )
            num_payment_attempts = random.randint(2, 5)
            amount = round(random.uniform(500, 5000), 2)  # Higher amounts for fraud
            chargeback_probability = random.uniform(0.3, 0.8)  # Higher chargeback probability

            # Multiple payment methods
            payment_methods = [self.generate_payment_method(True, user)
                             for _ in range(num_payment_attempts)]
        else:
            # Normal patterns
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.choice(self.normal_hours)
            )
            num_payment_attempts = 1
            amount = round(random.uniform(10, 1000), 2)
            chargeback_probability = random.uniform(0, 0.1)

            # Single payment method
            payment_methods = [self.generate_payment_method(False, user)]

        return {
            'transaction_id': str(uuid.uuid4()),
            'user_id': user['user_id'],
            'timestamp': timestamp.isoformat(),
            'amount': amount,
            'payment_methods': payment_methods,
            'num_payment_attempts': num_payment_attempts,
            'account_age_days': user['account_age_days'],
            'chargeback_probability': chargeback_probability,
            'is_unusual_time': timestamp.hour in self.unusual_hours,
            'is_high_amount': amount > 1000,
            'is_multiple_methods': len(payment_methods) > 1,
            'is_crypto': any(pm['method'] == 'crypto' for pm in payment_methods),
            'is_fraud': is_fraud
        }

    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset."""
        fraud_samples = int(self.num_samples * self.fraud_ratio)
        normal_samples = self.num_samples - fraud_samples

        # Generate transactions
        fraud_transactions = [self.generate_transaction(True) for _ in range(fraud_samples)]
        normal_transactions = [self.generate_transaction(False) for _ in range(normal_samples)]

        # Combine and shuffle
        all_transactions = fraud_transactions + normal_transactions
        random.shuffle(all_transactions)

        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)

        # Normalize payment methods
        df['primary_payment_method'] = df['payment_methods'].apply(lambda x: x[0]['method'])
        df['has_crypto_payment'] = df['payment_methods'].apply(
            lambda x: any(pm['method'] == 'crypto' for pm in x)
        )

        # Drop complex nested structures for ML readiness
        df = df.drop('payment_methods', axis=1)

        return df

def create_dataset(num_samples: int = 10000, fraud_ratio: float = 0.1,
                  output_file: str = 'payment_anomalies_dataset.csv'):
    """
    Create and save the payment anomalies dataset.

    Args:
        num_samples: Number of transactions to generate
        fraud_ratio: Ratio of fraudulent transactions
        output_file: Name of the output CSV file
    """
    generator = PaymentAnomalyDataset(num_samples, fraud_ratio)
    df = generator.generate_dataset()
    df.to_csv(output_file, index=False)
    print(f"Dataset generated with {num_samples} samples ({fraud_ratio*100}% fraud)")
    print(f"Dataset saved to {output_file}")
    return df

if __name__ == "__main__":
    create_dataset()
