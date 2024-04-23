import pandas as pd
import numpy as np


def create_dataset():
    np.random.seed(42)

    data = {
        'Operator': ['Operator A', 'Operator B', 'Operator C', 'Operator D'],
        'Speed': np.random.randint(20, 100, size=4),
        'Price': np.random.randint(30, 100, size=4),
        'Reliability': np.random.randint(1, 10, size=4)
    }

    telecom_dataset = pd.DataFrame(data)
    telecom_dataset.to_csv('variant_6.csv', index=False)


def normalize(df, columns):
    result = df.copy()
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name + '_normalized'] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def rank(data, weights):
    result = data.copy()
    for feature_name in data.columns[1:]:
        result[feature_name + '_rank'] = data[feature_name].rank(ascending=False) * weights[feature_name]
    result['Total_Rank'] = result[[col for col in result.columns if '_rank' in col]].sum(axis=1)
    return result

def evaluate_thresholds(data, thresholds):
    result = data.copy()
    for feature_name, threshold_value in thresholds.items():
        result[feature_name + '_pass'] = (result[feature_name] >= threshold_value).astype(int)
    return result

weights = {'Speed': 0.4, 'Price': 0.3, 'Reliability': 0.3}
thresholds = {'Speed': 70, 'Price': 40, 'Reliability': 5}

create_dataset()
df = pd.read_csv('variant_6.csv')
print(df)

normalized_data = normalize(df, ['Speed', 'Price'])
print(normalized_data)

ranked_data = rank(df, weights)
print(ranked_data[['Operator', 'Speed_rank', 'Price_rank', 'Reliability_rank', 'Total_Rank']])

threshold_results = evaluate_thresholds(df, thresholds)
print(threshold_results[['Operator', 'Speed_pass', 'Price_pass', 'Reliability_pass']])