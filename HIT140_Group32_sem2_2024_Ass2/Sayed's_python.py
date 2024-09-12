import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')


# 1: Summary Statistics for Each Column
def summary_statistics(df, dataset_name):
    print(f"\nSummary Statistics for {dataset_name}:")
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"\nColumn: {col}")
        print(f"Mean: {df[col].mean()}")
        print(f"Median: {df[col].median()}")
        print(f"Mode: {df[col].mode()[0]}")
        print(f"Standard Deviation: {df[col].std()}")
        print(f"Range: {df[col].max() - df[col].min()}")
        print(f"25th Percentile: {df[col].quantile(0.25)}")
        print(f"75th Percentile: {df[col].quantile(0.75)}")
        
summary_statistics(df1, 'Dataset 1')
summary_statistics(df2, 'Dataset 2')
summary_statistics(df3, 'Dataset 3')


# 2: Descriptive Analysis of Data Distributions
def describe_data_distributions(df, dataset_name):
    print(f"\nData Distribution Description for {dataset_name}:")
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"\nColumn: {col}")
        print(f"Minimum: {df[col].min()}")
        print(f"Maximum: {df[col].max()}")
        print(f"Mean: {df[col].mean()}")
        print(f"Median: {df[col].median()}")
        print(f"Standard Deviation: {df[col].std()}")
        print(f"Variance: {df[col].var()}")
        print(f"25th Percentile: {df[col].quantile(0.25)}")
        print(f"75th Percentile: {df[col].quantile(0.75)}")
        print(f"Number of Unique Values: {df[col].nunique()}")
        print(f"Most Frequent Value: {df[col].mode()[0]}")
        print(f"Frequency of Most Frequent Value: {df[col].value_counts().max()}")

describe_data_distributions(df1, 'Dataset 1')
describe_data_distributions(df2, 'Dataset 2')
describe_data_distributions(df3, 'Dataset 3')


# 3: Correlation Statistics
def correlation_statistics(df, dataset_name):
    print(f"\nCorrelation Statistics for {dataset_name}:")
    corr_matrix = df.corr()
    print(corr_matrix)

correlation_statistics(df1, 'Dataset 1')
correlation_statistics(df2, 'Dataset 2')
correlation_statistics(df3, 'Dataset 3')