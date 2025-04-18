import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_account_takeover_dataset(n_samples=50):
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='10T'),
        'user_id': np.random.randint(1000, 9999, n_samples),
        'ip_address': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
        'login_success': np.random.choice([1, 0], n_samples, p=[0.9, 0.1]),
        'login_attempts_1h': np.minimum(np.random.poisson(lam=2, size=n_samples), 5),
        'password_changed': np.random.choice([1, 0], n_samples, p=[0.05, 0.95]),
        'device_changed': np.random.choice([1, 0], n_samples, p=[0.1, 0.9]),
        'location_changed': np.random.choice([1, 0], n_samples, p=[0.15, 0.85])
    }

    df = pd.DataFrame(data)

    # Define suspicious behavior based on patterns:
    # 1. High login attempts with failed login
    # 2. Password change with location or device change
    # 3. Multiple changes (device, location, password) at once
    df['is_suspicious'] = 0

    # Rule 1: High login attempts (>3) with failed login
    df.loc[(df['login_attempts_1h'] > 3) & (df['login_success'] == 0), 'is_suspicious'] = 1

    # Rule 2: Password change with location or device change
    df.loc[(df['password_changed'] == 1) &
           ((df['location_changed'] == 1) | (df['device_changed'] == 1)),
           'is_suspicious'] = 1

    # Rule 3: Multiple changes at once (2 or more changes)
    changes_count = df['password_changed'] + df['device_changed'] + df['location_changed']
    df.loc[changes_count >= 2, 'is_suspicious'] = 1

    return df
