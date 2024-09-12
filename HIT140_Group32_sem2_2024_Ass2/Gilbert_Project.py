#The Impact of Screen Time on Well-Being: Analyzing Optimism Levels and Statistical Differences 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, norm, ttest_ind

# Load datasets
df1 = pd.read_csv('dataset1.csv')  # Load demographic information
df2 = pd.read_csv('dataset2.csv')  # Load digital screen time information
df3 = pd.read_csv('dataset3.csv')  # Load well-being indicators

# Merge datasets on the unique ID
df_merged = pd.merge(pd.merge(df1, df2, on='ID'), df3, on='ID')

# Calculate average screen time by averaging computer, video game, smartphone, and TV usage on weekdays
df_merged['Average_Screen_Time'] = df_merged[['C_wk', 'G_wk', 'S_wk', 'T_wk']].mean(axis=1)

# Define high and low screen time groups based on median screen time
median_screen_time = df_merged['Average_Screen_Time'].median()  # Compute the median of average screen time
df_merged['Screen_Time_Group'] = np.where(df_merged['Average_Screen_Time'] > median_screen_time, 'High', 'Low')  # Classify as 'High' or 'Low'

# Perform Independent Samples T-Test to compare well-being between high and low screen time groups
high_screen_time = df_merged[df_merged['Screen_Time_Group'] == 'High']['Optm']  # Extract optimism scores for high screen time group
low_screen_time = df_merged[df_merged['Screen_Time_Group'] == 'Low']['Optm']  # Extract optimism scores for low screen time group
t_stat, p_val = ttest_ind(high_screen_time, low_screen_time)  # Perform t-test

# Output T-Statistic and P-Value
print(f'T-Statistic: {t_stat}')
print(f'P-Value: {p_val}')

# Test the hypothesis at alpha = 0.05 significance level
alpha = 0.05
if p_val < alpha:
    print("Reject the null hypothesis: Significant difference in well-being between groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference in well-being between groups.")

# Calculate Z-Scores for the well-being indicator in both groups
mean_high = high_screen_time.mean()  # Mean of well-being scores in high screen time group
std_high = high_screen_time.std()  # Standard deviation of well-being scores in high screen time group
z_scores_high = (high_screen_time - mean_high) / std_high  # Z-Scores for high screen time group

mean_low = low_screen_time.mean()  # Mean of well-being scores in low screen time group
std_low = low_screen_time.std()  # Standard deviation of well-being scores in low screen time group
z_scores_low = (low_screen_time - mean_low) / std_low  # Z-Scores for low screen time group

# Function to compute confidence intervals
def compute_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)  # Calculate the mean of the data
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))  # Calculate the standard error
    margin_of_error = std_err * t.ppf((1 + confidence) / 2., len(data) - 1)  # Compute margin of error
    return mean - margin_of_error, mean + margin_of_error  # Return the confidence interval

# Calculate Confidence Intervals for high and low screen time groups
conf_interval_high = compute_confidence_interval(high_screen_time)
conf_interval_low = compute_confidence_interval(low_screen_time)

# Visualization
plt.figure(figsize=(18, 12))  # Set the figure size

# Line Graph of Well-Being Indicator by Screen Time Group
plt.subplot(2, 2, 1)  # Create a subplot in a 2x2 grid, position 1
mean_wellbeing = df_merged.groupby('Screen_Time_Group')['Optm'].mean().reset_index()  # Compute mean well-being for each group
plt.plot(mean_wellbeing['Screen_Time_Group'], mean_wellbeing['Optm'], marker='o', linestyle='-', color='b')  # Plot the means
plt.title('Mean Well-Being (Optimism) by Screen Time Group')  # Title of the plot
plt.xlabel('Screen Time Group')  # X-axis label
plt.ylabel('Mean Feeling Optimistic')  # Y-axis label
plt.grid(True)  # Show grid
plt.xticks(mean_wellbeing['Screen_Time_Group'])  # Set x-axis ticks

# Line Graph of t-Distribution and Standard Normal Distribution
plt.subplot(2, 2, 2)  # Create a subplot in a 2x2 grid, position 2
x = np.linspace(-4, 4, 100)  # Generate x values for plotting
df = len(df_merged) - 2  # Degrees of freedom for t-distribution
plt.plot(x, t.pdf(x, df=df), label=f't-distribution (df={df})', color='blue')  # Plot t-distribution
plt.plot(x, norm.pdf(x), label='Standard Normal Distribution', color='red', linestyle='--')  # Plot standard normal distribution
plt.title('t-Distribution vs Standard Normal Distribution')  # Title of the plot
plt.xlabel('Score')  # X-axis label
plt.ylabel('Probability Density')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Show grid

# Line Graph of Z-Scores Distribution
plt.subplot(2, 2, 3)  # Create a subplot in a 2x2 grid, position 3
sns.kdeplot(z_scores_high, label='High Screen Time Z-Scores', color='blue', linestyle='-', fill=True)  # KDE plot for high screen time z-scores
sns.kdeplot(z_scores_low, label='Low Screen Time Z-Scores', color='green', linestyle='--', fill=True)  # KDE plot for low screen time z-scores
plt.title('Density of Z-Scores')  # Title of the plot
plt.xlabel('Z-Score')  # X-axis label
plt.ylabel('Density')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Show grid

# Line Graph of Confidence Intervals
plt.subplot(2, 2, 4)  # Create a subplot in a 2x2 grid, position 4
x_labels = ['High Screen Time', 'Low Screen Time']  # X-axis labels
means = [mean_high, mean_low]  # Means of well-being scores
conf_intervals = [conf_interval_high[1] - conf_interval_high[0], conf_interval_low[1] - conf_interval_low[0]]  # Width of confidence intervals
plt.plot(x_labels, means, marker='o', color='b', linestyle='-', label='Mean Well-Being')  # Plot means
plt.errorbar(x_labels, means, yerr=np.array(conf_intervals) / 2, fmt='o', color='b', capsize=5)  # Add error bars for confidence intervals
plt.title('Mean Well-Being with Confidence Intervals')  # Title of the plot
plt.xlabel('Screen Time Group')  # X-axis label
plt.ylabel('Mean Well-Being (Optimism)')  # Y-axis label
plt.grid(True)  # Show grid

plt.tight_layout()  # Adjust subplot parameters to give some padding
plt.show()  # Display all plots

# Print summary statistics and Z-Scores
print("\nHigh Screen Time Group Summary:")
print(f"Mean: {mean_high}, Std Dev: {std_high}")  # Print mean and standard deviation
print("Sample Z-Scores:", z_scores_high.head())  # Print sample z-scores

print("\nLow Screen Time Group Summary:")
print(f"Mean: {mean_low}, Std Dev: {std_low}")  # Print mean and standard deviation
print("Sample Z-Scores:", z_scores_low.head())  # Print sample z-scores

print("\nConfidence Intervals:")
print(f"High Screen Time Confidence Interval: {conf_interval_high}")  # Print confidence interval for high screen time
print(f"Low Screen Time Confidence Interval: {conf_interval_low}")  # Print confidence interval for low screen time
