import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets_02_20_2025.features.AccountTakeover import generate_account_takeover_dataset

account_takeover_df = generate_account_takeover_dataset(n_samples=50)

# Display basic information about the dataset
print("\nDataset Info:")
print(account_takeover_df.info())

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(account_takeover_df.head())

# Save the dataset to a CSV file
output_path = "account_takeover_dataset.csv"
account_takeover_df.to_csv(output_path, index=False)
print(f"\nDataset saved to: {output_path}")

# Display some basic statistics
print("\nBasic statistics:")
print(f"Total number of records: {len(account_takeover_df)}")
print(f"Number of suspicious activities: {account_takeover_df['is_suspicious'].sum()}")
print(f"Percentage of suspicious activities: {(account_takeover_df['is_suspicious'].mean() * 100):.2f}%")
