import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Now import after adding to path
from datasets_02_20_2025.features.DeviceFingerprint import generate_device_fingerprints_dataset

# Generate the dataset
device_fingerprints_df = generate_device_fingerprints_dataset(n_samples=50)

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, 'datasets_02_20_2025', 'generated_datasets')
os.makedirs(output_dir, exist_ok=True)

# Display basic information about the dataset
print("\nDataset Info:")
print(device_fingerprints_df.info())

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(device_fingerprints_df.head())

# Save the dataset to a CSV file
output_path = os.path.join(output_dir, 'device_fingerprint_dataset.csv')
device_fingerprints_df.to_csv(output_path, index=False)
print(f"\nDataset saved to: {output_path}")

# Basic statistics
print("\nBasic Statistics:")
print(f"Total number of records: {len(device_fingerprints_df)}")
print(f"Number of suspicious activities: {device_fingerprints_df['is_suspicious'].sum()}")
print(f"Percentage of suspicious activities: {(device_fingerprints_df['is_suspicious'].mean() * 100):.2f}%")

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: OS Distribution
plt.subplot(2, 2, 1)
sns.countplot(data=device_fingerprints_df, x='os', hue='is_suspicious')
plt.title('OS Distribution by Suspicious Activity')
plt.xticks(rotation=45)

# Plot 2: Browser Distribution
plt.subplot(2, 2, 2)
sns.countplot(data=device_fingerprints_df, x='browser', hue='is_suspicious')
plt.title('Browser Distribution by Suspicious Activity')
plt.xticks(rotation=45)

# Plot 3: Suspicious vs Non-suspicious
plt.subplot(2, 2, 3)
suspicious_counts = device_fingerprints_df['is_suspicious'].value_counts()
plt.pie(suspicious_counts, labels=['Normal', 'Suspicious'], autopct='%1.1f%%')
plt.title('Distribution of Suspicious Activities')

# Plot 4: OS-Browser Compatibility Heatmap
plt.subplot(2, 2, 4)
os_browser_pairs = pd.crosstab(device_fingerprints_df['os'], device_fingerprints_df['browser'])
sns.heatmap(os_browser_pairs, cmap='YlOrRd')
plt.title('OS-Browser Pairs')

plt.tight_layout()

# Save visualizations
plt.savefig(os.path.join(output_dir, 'device_fingerprint_analysis.png'))
print(f"Visualizations saved to: {output_dir}")

# Print detailed analysis
print("\nDetailed Analysis:")
print("\nOS Distribution:")
print(device_fingerprints_df['os'].value_counts())

print("\nBrowser Distribution:")
print(device_fingerprints_df['browser'].value_counts())

print("\nScreen Resolution Distribution:")
print(device_fingerprints_df['screen_resolution'].value_counts().head())

# Analyze suspicious patterns
suspicious_df = device_fingerprints_df[device_fingerprints_df['is_suspicious'] == 1]
print("\nSuspicious Pattern Analysis:")
if not suspicious_df.empty:
    print(f"\nMost common incompatible combinations:")
    print("\nOS-Browser combinations in suspicious cases:")
    print(pd.crosstab(suspicious_df['os'], suspicious_df['browser']))
    print("\nOS-Resolution combinations in suspicious cases:")
    print(pd.crosstab(suspicious_df['os'], suspicious_df['screen_resolution']).head())
else:
    print("No suspicious cases found in the dataset")
