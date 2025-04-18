import pandas as pd
import numpy as np
from faker import Faker

def generate_email_similarity_dataset(n_samples=50):
    fake = Faker()

    legitimate_domains = [
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'aol.com', 'protonmail.com', 'icloud.com', 'mail.com',
        'zoho.com', 'fastmail.com', 'tutanota.com', 'gmx.com',
        'yandex.com', 'pm.me', 'live.com', 'msn.com'
    ]

    def generate_similar_email(email):
        username, domain = email.split('@')
        variations = [
            f"{username}1@{domain}",
            f"{username}.{np.random.randint(100)}@{domain}",
            f"{username}_{np.random.randint(100)}@{domain}",
            f"{username}{np.random.randint(100)}@{domain}",
            f"{username}.{fake.word()}@{domain}",
            f"{username}_{fake.word()}@{domain}",
            username.replace('o', '0').replace('i', '1').replace('l', '1') + f"@{domain}"
        ]
        return np.random.choice(variations)

    data = {
        'original_email': [f"{fake.user_name()}@{np.random.choice(legitimate_domains)}" for _ in range(n_samples)],
        'registration_email': [],
        'email_age_days': np.random.randint(1, 365, n_samples),
        'email_provider': np.random.choice(legitimate_domains, n_samples),
        'account_creation_hour': np.random.randint(0, 24, n_samples),
        'ip_velocity': np.random.poisson(lam=2, size=n_samples),  # Number of IPs used
        'failed_login_attempts': np.random.poisson(lam=1, size=n_samples),
        'profile_completeness': np.random.choice([1, 0], n_samples, p=[0.8, 0.2])
    }

    # Generate registration emails based on patterns
    for i in range(n_samples):
        # Determine if this should be a suspicious case based on multiple factors
        is_suspicious_pattern = (
            (data['email_age_days'][i] < 7 and  # New account
             data['ip_velocity'][i] > 3) or  # Multiple IPs
            (data['failed_login_attempts'][i] > 3 and  # Many failed logins
             data['profile_completeness'][i] == 0) or  # Incomplete profile
            (data['account_creation_hour'][i] in [2, 3, 4] and  # Created during suspicious hours
             data['ip_velocity'][i] > 2)  # With multiple IPs
        )

        if is_suspicious_pattern:
            data['registration_email'].append(generate_similar_email(data['original_email'][i]))
        else:
            data['registration_email'].append(data['original_email'][i])

    # Create DataFrame
    df = pd.DataFrame(data)

    # Initialize is_suspicious column
    df['is_suspicious'] = 0

    # Rule 1: Similar but different email with new account
    df.loc[(df['registration_email'] != df['original_email']) &
           (df['email_age_days'] < 7), 'is_suspicious'] = 1

    # Rule 2: High IP velocity with similar email
    df.loc[(df['ip_velocity'] > 3) &
           (df['registration_email'] != df['original_email']), 'is_suspicious'] = 1

    # Rule 3: Multiple failed login attempts with similar email
    df.loc[(df['failed_login_attempts'] > 3) &
           (df['registration_email'] != df['original_email']), 'is_suspicious'] = 1

    # Rule 4: Account created during suspicious hours (2-4 AM) with similar email
    df.loc[(df['account_creation_hour'].isin([2, 3, 4])) &
           (df['registration_email'] != df['original_email']), 'is_suspicious'] = 1

    # Rule 5: Incomplete profile with similar email
    df.loc[(df['profile_completeness'] == 0) &
           (df['registration_email'] != df['original_email']), 'is_suspicious'] = 1

    return df
