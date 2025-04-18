import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets_02_20_2025.features.BillingRegion import generate_billing_region_dataset

# Generate the dataset
billing_region_df = generate_billing_region_dataset(n_samples=50)

# Display basic information about the dataset
print("\nDataset Info:")
print(billing_region_df.info())

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(billing_region_df.head())

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'generated_datasets')
os.makedirs(output_dir, exist_ok=True)
