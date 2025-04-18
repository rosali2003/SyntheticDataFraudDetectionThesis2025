import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Fix the import to match the correct function name
from datasets_02_20_2025.features.EmailSimilarity import generate_email_similarity_dataset

# Generate the dataset
email_similarity_df = generate_email_similarity_dataset(n_samples=50)

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, 'datasets_02_20_2025', 'generated_datasets')
os.makedirs(output_dir, exist_ok=True)

# Display basic information about the dataset
print("\nDataset Info:")
print(email_similarity_df.info())

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(email_similarity_df.head())

# Save the dataset to a CSV file
output_path = os.path.join(output_dir, 'email_similarity_dataset.csv')
email_similarity_df.to_csv(output_path, index=False)


