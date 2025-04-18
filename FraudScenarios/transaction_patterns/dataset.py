import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import uuid

class TransactionPatternDataset:
    def __init__(self, num_samples: int = 10000, fraud_ratio: float = 0.1):
        """
        Initialize the dataset generator for transaction pattern fraud scenarios.

        Args:
            num_samples: Total number of transactions to generate
            fraud_ratio: Ratio of fraudulent transactions (default 10%)
        """
        self.num_samples = num_samples
        self.fraud_ratio = fraud_ratio
        self.faker = Faker()

        # Product categories and their typical price ranges
        self.product_categories = {
            'electronics': (100, 2000),
            'clothing': (20, 200),
            'accessories': (10, 100),
            'home_goods': (30, 500),
            'digital_goods': (5, 100)
        }

        # Flash sale details
        self.flash_sale_items = {
            'limited_edition_phone': 999.99,
            'designer_bag': 499.99,
            'gaming_console': 299.99,
            'premium_headphones': 199.99
        }

        # Initialize user pool for consistent patterns
        self.user_pool = self._create_user_pool(int(num_samples * 0.1))

    def _create_user_pool(self, num_users: int) -> List[Dict]:
        """Create a pool of users with their shopping patterns."""
        return [{
            'user_id': str(uuid.uuid4()),
            'email': self.faker.email(),
            'account_age_days': random.randint(1, 365),
            'avg_transaction_value': random.uniform(50, 500),
            'typical_categories': random.sample(list(self.product_categories.keys()),
                                             random.randint(1, 3)),
            'payment_methods': random.sample(['credit_card', 'debit_card', 'digital_wallet'],
                                          random.randint(1, 3))
        } for _ in range(num_users)]

    def generate_cart_items(self, is_fraud: bool, user: Dict) -> List[Dict]:
        """Generate cart items based on normal or fraudulent patterns."""
        if is_fraud:
            if random.random() < 0.3:  # High-value single item fraud
                category = random.choice(list(self.product_categories.keys()))
                min_price, max_price = self.product_categories[category]
                return [{
                    'category': category,
                    'quantity': 1,
                    'unit_price': round(random.uniform(max_price * 0.8, max_price * 1.2), 2)
                }]
            else:  # Multiple small items fraud
                items = []
                num_items = random.randint(5, 15)
                for _ in range(num_items):
                    category = random.choice(list(self.product_categories.keys()))
                    min_price, max_price = self.product_categories[category]
                    items.append({
                        'category': category,
                        'quantity': random.randint(1, 3),
                        'unit_price': round(random.uniform(min_price, min_price * 1.5), 2)
                    })
                return items
        else:
            items = []
            num_items = random.randint(1, 5)
            for _ in range(num_items):
                category = random.choice(user['typical_categories'])
                min_price, max_price = self.product_categories[category]
                items.append({
                    'category': category,
                    'quantity': random.randint(1, 3),
                    'unit_price': round(random.uniform(min_price, max_price), 2)
                })
            return items

    def generate_transaction_sequence(self, is_fraud: bool, user: Dict) -> List[Dict]:
        """Generate a sequence of related transactions."""
        base_timestamp = datetime.now() - timedelta(days=random.randint(0, 30))

        if is_fraud:
            # Generate testing transactions followed by fraud
            num_test_transactions = random.randint(2, 5)
            transactions = []

            # Small testing transactions
            for i in range(num_test_transactions):
                timestamp = base_timestamp + timedelta(minutes=random.randint(5, 30) * i)
                transactions.append({
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': user['user_id'],
                    'timestamp': timestamp.isoformat(),
                    'cart_items': [{'category': 'digital_goods', 'quantity': 1,
                                  'unit_price': round(random.uniform(5, 20), 2)}],
                    'payment_method': random.choice(user['payment_methods']),
                    'is_test_transaction': True
                })

            # Fraudulent transaction
            timestamp = base_timestamp + timedelta(minutes=random.randint(30, 60))
            transactions.append({
                'transaction_id': str(uuid.uuid4()),
                'user_id': user['user_id'],
                'timestamp': timestamp.isoformat(),
                'cart_items': self.generate_cart_items(True, user),
                'payment_method': 'new_card',  # New payment method for fraud
                'is_test_transaction': False
            })

            return transactions
        else:
            # Generate normal transaction sequence
            num_transactions = random.randint(1, 3)
            return [{
                'transaction_id': str(uuid.uuid4()),
                'user_id': user['user_id'],
                'timestamp': (base_timestamp + timedelta(days=random.randint(0, 7))).isoformat(),
                'cart_items': self.generate_cart_items(False, user),
                'payment_method': random.choice(user['payment_methods']),
                'is_test_transaction': False
            } for _ in range(num_transactions)]

    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset with transaction patterns."""
        fraud_samples = int(self.num_samples * self.fraud_ratio)
        normal_samples = self.num_samples - fraud_samples

        all_transactions = []

        # Generate fraud patterns
        for _ in range(fraud_samples):
            user = random.choice(self.user_pool)
            transactions = self.generate_transaction_sequence(True, user)
            all_transactions.extend(transactions)

        # Generate normal patterns
        for _ in range(normal_samples):
            user = random.choice(self.user_pool)
            transactions = self.generate_transaction_sequence(False, user)
            all_transactions.extend(transactions)

        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)

        # Calculate derived features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['total_amount'] = df['cart_items'].apply(
            lambda x: sum(item['quantity'] * item['unit_price'] for item in x)
        )
        df['num_items'] = df['cart_items'].apply(len)
        df['is_high_value'] = df['total_amount'] > 1000
        df['is_new_payment_method'] = df['payment_method'] == 'new_card'

        # Calculate transaction velocity features
        df = df.sort_values('timestamp')
        df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['transactions_last_hour'] = df.groupby('user_id')['timestamp'].transform(
            lambda x: x.rolling('1H').count()
        )

        # Mark fraud based on patterns
        df['is_fraud'] = (
            (df['is_test_transaction']) |  # Testing transactions
            (df['is_high_value'] & df['is_new_payment_method']) |  # High value with new payment
            (df['transactions_last_hour'] > 5)  # High velocity
        )

        # Drop intermediate columns and complex structures
        df = df.drop(['cart_items', 'is_test_transaction'], axis=1)

        return df

def create_dataset(num_samples: int = 10000, fraud_ratio: float = 0.1,
                  output_file: str = 'transaction_patterns_dataset.csv'):
    """
    Create and save the transaction patterns dataset.

    Args:
        num_samples: Number of transactions to generate
        fraud_ratio: Ratio of fraudulent transactions
        output_file: Name of the output CSV file
    """
    generator = TransactionPatternDataset(num_samples, fraud_ratio)
    df = generator.generate_dataset()
    df.to_csv(output_file, index=False)
    print(f"Dataset generated with {len(df)} transactions")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    print(f"Dataset saved to {output_file}")
    return df

if __name__ == "__main__":
    create_dataset()
