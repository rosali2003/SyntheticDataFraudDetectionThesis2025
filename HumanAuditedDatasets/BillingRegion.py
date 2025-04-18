import pandas as pd
import numpy as np

def generate_billing_region_dataset(n_samples=50):
    countries = ['US', 'UK', 'CA', 'AU', 'FR', 'DE', 'JP', 'BR', 'RU', 'IN']

    data = {
        'transaction_id': range(1, n_samples + 1),
        'user_id': np.random.randint(1000, 9999, n_samples),
        'ip_country': np.random.choice(countries, n_samples),
        'billing_country': [],
        'shipping_country': [],
        'card_issuing_country': [],
        'transaction_amount': np.random.uniform(10, 1000, n_samples)
    }

    for i in range(n_samples):
        ip_country = data['ip_country'][i]

        has_billing_mismatch = np.random.choice([0, 1], p=[0.8, 0.2])
        has_shipping_mismatch = np.random.choice([0, 1], p=[0.85, 0.15])
        has_card_mismatch = np.random.choice([0, 1], p=[0.9, 0.1])

        # Set countries based on mismatches
        data['billing_country'].append(
            np.random.choice(countries) if has_billing_mismatch else ip_country
        )
        data['shipping_country'].append(
            np.random.choice(countries) if has_shipping_mismatch else ip_country
        )
        data['card_issuing_country'].append(
            np.random.choice(countries) if has_card_mismatch else ip_country
        )

    df = pd.DataFrame(data)

    df['is_suspicious'] = 0

    # Rule 1: Different billing and IP country with high transaction amount
    df.loc[(df['billing_country'] != df['ip_country']) &
           (df['transaction_amount'] > 500), 'is_suspicious'] = 1

    # Rule 2: All countries different from IP country
    df.loc[(df['billing_country'] != df['ip_country']) &
           (df['shipping_country'] != df['ip_country']) &
           (df['card_issuing_country'] != df['ip_country']), 'is_suspicious'] = 1

    # Rule 3: Card issuing country different from billing country with high amount
    df.loc[(df['card_issuing_country'] != df['billing_country']) &
           (df['transaction_amount'] > 750), 'is_suspicious'] = 1

    # Rule 4: Three different countries for billing, shipping, and card
    df.loc[(df['billing_country'] != df['shipping_country']) &
           (df['shipping_country'] != df['card_issuing_country']) &
           (df['billing_country'] != df['card_issuing_country']), 'is_suspicious'] = 1

    return df
