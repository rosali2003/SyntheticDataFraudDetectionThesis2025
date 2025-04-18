import pandas as pd
import numpy as np
from faker import Faker

def generate_transaction_email_dataset(n_samples=50):
    fake = Faker()

    legitimate_domains = [
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'aol.com', 'protonmail.com', 'icloud.com', 'mail.com',
        'zoho.com', 'fastmail.com', 'tutanota.com', 'gmx.com',
        'yandex.com', 'pm.me', 'live.com', 'msn.com'
    ]

    data = {
        'transaction_id': range(1, n_samples + 1),
        'user_id': np.random.randint(1000, 9999, n_samples),
        'transaction_amount': np.random.uniform(10, 1000, n_samples),
        'transaction_email': [],
        'account_email': [],
        'email_age_days': np.random.randint(1, 365, n_samples),
        'previous_transactions_count': np.random.poisson(lam=5, size=n_samples),
        'transaction_time': pd.date_range(start='2024-01-01', periods=n_samples, freq='10T'),
        'email_verified': np.random.choice([1, 0], n_samples, p=[0.9, 0.1]),
        'transaction_country': np.random.choice(['US', 'UK', 'CA', 'AU', 'FR'], n_samples)
    }

    for i in range(n_samples):
        account_domain = np.random.choice(legitimate_domains)
        account_username = fake.user_name()
        account_email = f"{account_username}@{account_domain}"
        data['account_email'].append(account_email)

        if (data['email_age_days'][i] < 7 and
            data['previous_transactions_count'][i] < 2 and
            data['transaction_amount'][i] > 500):
            trans_domain = np.random.choice(legitimate_domains)
            trans_username = fake.user_name()
            while trans_username == account_username:
                trans_username = fake.user_name()
            transaction_email = f"{trans_username}@{trans_domain}"
        else:
            transaction_email = account_email

        data['transaction_email'].append(transaction_email)

    df = pd.DataFrame(data)

    df['is_suspicious'] = 0

    # Rule 1: Different transaction email from account email
    df.loc[df['transaction_email'] != df['account_email'], 'is_suspicious'] = 1

    # Rule 2: New email with high value transaction
    df.loc[(df['email_age_days'] < 7) &
           (df['transaction_amount'] > 500), 'is_suspicious'] = 1

    # Rule 3: Unverified email with significant transaction history
    df.loc[(df['email_verified'] == 0) &
           (df['previous_transactions_count'] > 5), 'is_suspicious'] = 1

    # Rule 4: Multiple transactions in short time with different email
    df['time_diff'] = df['transaction_time'].diff()
    df.loc[(df['time_diff'].dt.total_seconds() < 3600) &  # Less than 1 hour
           (df['transaction_email'] != df['account_email']) &
           (df['transaction_amount'] > 200), 'is_suspicious'] = 1

    # Rule 5: New email with no transaction history
    df.loc[(df['email_age_days'] < 30) &
           (df['previous_transactions_count'] == 0) &
           (df['transaction_amount'] > 300), 'is_suspicious'] = 1

    return df
