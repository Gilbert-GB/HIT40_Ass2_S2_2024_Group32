import pandas as pd
from scipy import stats

# Load the datasets
dataset1 = pd.read_csv('C:/Users/sinhp/HIT 140/dataset1.csv')
dataset2 = pd.read_csv('C:/Users/sinhp/HIT 140/dataset2.csv')
dataset3 = pd.read_csv('C:/Users/sinhp/HIT 140/dataset3.csv')

# Merge the datasets
data_merged = pd.merge(dataset1, dataset2, on='ID')
data_merged = pd.merge(data_merged, dataset3, on='ID')

# Well-being indicators
wellbeing_indicators = ['Optm', 'Usef', 'Relx']
for indicator in wellbeing_indicators:
    t_statistic, p_value = stats.ttest_ind(
        data_merged[indicator][data_merged['deprived'] == 0],
        data_merged[indicator][data_merged['deprived'] == 1]
    )
    interpretation = interpret_t_test(t_statistic, p_value)
    print(f"{indicator}: {interpretation}")

# Screen time activities
screen_time_activities = ['C_we', 'C_wk', 'G_we', 'G_wk']
for activity in screen_time_activities:
    t_statistic, p_value = stats.ttest_ind(
        data_merged[activity][data_merged['deprived'] == 0],
        data_merged[activity][data_merged['deprived'] == 1]
    )
    interpretation = interpret_t_test(t_statistic, p_value)
    print(f"{activity}: {interpretation}")

# Function to interpret t-value and p-value
def interpret_t_test(t_statistic, p_value, group1_label="Non-Deprived", group2_label="Deprived"):
    if p_value < 0.05:
        if t_statistic < 0:
            return f"{group2_label} has a significantly lower mean than {group1_label} (t = {t_statistic:.2f}, p = {p_value:.4e})"
        else:
            return f"{group2_label} has a significantly higher mean than {group1_label} (t = {t_statistic:.2f}, p = {p_value:.4e})"
    else:
        return f"No significant difference between {group1_label} and {group2_label} (t = {t_statistic:.2f}, p = {p_value:.4e})"