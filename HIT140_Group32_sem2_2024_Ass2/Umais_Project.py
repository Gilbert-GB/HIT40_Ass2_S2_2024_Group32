import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
df1 = pd.read_csv('dataset1.csv') 
df2 = pd.read_csv('dataset2.csv')  
df3 = pd.read_csv('dataset3.csv')  

#1: Descriptive Statistics for Numerical Data
print("Dataset 1 Descriptive Statistics:")
print(df1.describe())

print("\nDataset 2 Descriptive Statistics:")
print(df2.describe())

print("\nDataset 3 Descriptive Statistics:")
print(df3.describe())

# Summary statistics with NumPy
print("\nAdditional Summary (Dataset 1) with NumPy:")
for col in df1.select_dtypes(include=[np.number]).columns:
    print(f"\nColumn: {col}")
    print(f"Mean: {np.mean(df1[col])}")
    print(f"Median: {np.median(df1[col].dropna())}")  # dropna() to avoid NaNs
    print(f"Standard Deviation: {np.std(df1[col])}")
    print(f"Variance: {np.var(df1[col])}")

print("\nAdditional Summary (Dataset 2) with NumPy:")
for col in df2.select_dtypes(include=[np.number]).columns:
    print(f"\nColumn: {col}")
    print(f"Mean: {np.mean(df2[col])}")
    print(f"Median: {np.median(df2[col].dropna())}")  # dropna() to avoid NaNs
    print(f"Standard Deviation: {np.std(df2[col])}")
    print(f"Variance: {np.var(df2[col])}")

print("\nAdditional Summary (Dataset 2) with NumPy:")
for col in df3.select_dtypes(include=[np.number]).columns:
    print(f"\nColumn: {col}")
    print(f"Mean: {np.mean(df3[col])}")
    print(f"Median: {np.median(df3[col].dropna())}")  # dropna() to avoid NaNs
    print(f"Standard Deviation: {np.std(df3[col])}")
    print(f"Variance: {np.var(df3[col])}")


#2: Checking for Missing Values
print("\nMissing Values in Dataset 1:")
print(df1.isnull().sum())

print("\nMissing Values in Dataset 2:")
print(df2.isnull().sum())

print("\nMissing Values in Dataset 3:")
print(df3.isnull().sum())

# Checking the first few rows of datasets
print("\nFirst 5 rows of Dataset 1:")
print(df1.head())

print("\nFirst 5 rows of Dataset 2:")
print(df2.head())

print("\nFirst 5 rows of Dataset 3:")
print(df3.head())

# Identifying the data types of each column
print("\nData Types in Dataset 1:")
print(df1.dtypes)

print("\nData Types in Dataset 2:")
print(df2.dtypes)

print("\nData Types in Dataset 3:")
print(df3.dtypes)

# Visualization with Matplotlib
# Plot histograms for numerical columns in Dataset 1
df1_numeric = df1.select_dtypes(include=[np.number])

for col in df1_numeric.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df1[col].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col} (Dataset 1)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Plot histograms for numerical columns in Dataset 2
df2_numeric = df2.select_dtypes(include=[np.number])

for col in df2_numeric.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df2[col].dropna(), bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Histogram of {col} (Dataset 2)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Plot histograms for numerical columns in Dataset 3
df3_numeric = df3.select_dtypes(include=[np.number])

for col in df3_numeric.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df3[col].dropna(), bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Histogram of {col} (Dataset 3)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()