import pandas as pd
import numpy as np
from faker import Faker

def generate_disposable_email_dataset(n_samples=50):
    fake = Faker()

    disposable_domains = [
        'tempmail.com', 'throwaway.com', '10minutemail.com',
        'guerrillamail.com', 'mailinator.com', 'temporary-mail.net',
        'disposable.com', 'yopmail.com', 'trashmail.com',
        'sharklasers.com', 'getairmail.com', 'temp-mail.org',
        'fakeinbox.com', 'tempinbox.com', 'emailondeck.com',
        'burnermail.io', 'temp-mail.ru', 'dispostable.com'
    ]

    legitimate_domains = [
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'aol.com', 'protonmail.com', 'icloud.com', 'mail.com',
        'zoho.com', 'fastmail.com', 'tutanota.com', 'gmx.com',
        'yandex.com', 'pm.me', 'live.com', 'msn.com',
        'edu.com', 'student.edu', 'university.edu', 'alumni.edu',
        'qq.com', '163.com', 'sina.com', 'mail.ru',
        'web.de', 'orange.fr', 'free.fr', 'btinternet.com',
        'company.com', 'corporation.com', 'business.com', 'enterprise.com'
    ]

    data = {
        'email': [],
        'domain': [],
        'registration_date': pd.date_range(start='2024-01-01', periods=n_samples, freq='10T'),
        'email_confirmed': np.random.choice([1, 0], n_samples, p=[0.8, 0.2]),
        'account_age_days': np.random.randint(1, 365, n_samples),
        'login_count': np.random.poisson(lam=10, size=n_samples),
        'has_recovery_email': np.random.choice([1, 0], n_samples, p=[0.7, 0.3]),
        'profile_completed': np.random.choice([1, 0], n_samples, p=[0.75, 0.25])
    }

    for i in range(n_samples):
        if (data['login_count'][i] < 2 and
            data['account_age_days'][i] < 7 and
            data['has_recovery_email'][i] == 0 and
            data['profile_completed'][i] == 0):
            domain = np.random.choice(disposable_domains)
        else:
            domain = np.random.choice(legitimate_domains)

        username = fake.user_name()
        email = f"{username}@{domain}"

        data['email'].append(email)
        data['domain'].append(domain)

    df = pd.DataFrame(data)

    df['is_suspicious'] = 0

    # Rule 1: Known disposable email domains
    df.loc[df['domain'].isin(disposable_domains), 'is_suspicious'] = 1

    # Rule 2: New account with low engagement
    df.loc[(df['account_age_days'] < 7) &
           (df['login_count'] < 3) &
           (df['profile_completed'] == 0), 'is_suspicious'] = 1

    # Rule 3: Unconfirmed email with no recovery
    df.loc[(df['email_confirmed'] == 0) &
           (df['has_recovery_email'] == 0), 'is_suspicious'] = 1

    # Rule 4: Very new account with minimal setup
    df.loc[(df['account_age_days'] < 2) &
           (df['profile_completed'] == 0) &
           (df['has_recovery_email'] == 0), 'is_suspicious'] = 1

    return df
