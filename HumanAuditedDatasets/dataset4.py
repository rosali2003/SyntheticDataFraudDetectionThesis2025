import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Now import after adding to path
from datasets_02_20_2025.features.DisposableEmail import generate_transaction_email_dataset

# Generate the dataset
disposable_emails_df = generate_transaction_email_dataset(n_samples=50)

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, 'datasets_02_20_2025', 'generated_datasets')
os.makedirs(output_dir, exist_ok=True)

# Display basic information about the dataset
print("\nDataset Info:")
print(disposable_emails_df.info())

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(disposable_emails_df.head())

# Save the dataset to a CSV file
output_path = os.path.join(output_dir, 'disposable_email_dataset.csv')
disposable_emails_df.to_csv(output_path, index=False)

