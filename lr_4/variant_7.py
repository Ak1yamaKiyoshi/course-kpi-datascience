import pandas as pd
import numpy as np

def create_mobile_dataset():
    np.random.seed(42)

    data = {
        'Operator': ['Operator X', 'Operator Y', 'Operator Z', 'Operator W'],
        'Speed': np.random.randint(10, 80, size=4),
        'Price': np.random.randint(40, 120, size=4),
        'Reliability': np.random.randint(5, 15, size=4)
    }

    mobile_dataset = pd.DataFrame(data)
    mobile_dataset.to_csv('variant_7.csv', index=False)

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

weights_mobile = {'Speed': 0.4, 'Price': 0.3, 'Reliability': 0.3}
thresholds_mobile = {'Speed': 50, 'Price': 60, 'Reliability': 7}

create_mobile_dataset()
df_mobile = pd.read_csv('variant_7.csv')
print(df_mobile)

normalized_data_mobile = normalize(df_mobile, ['Speed', 'Price'])
print(normalized_data_mobile)

ranked_data_mobile = rank(df_mobile, weights_mobile)
print(ranked_data_mobile[['Operator', 'Speed_rank', 'Price_rank', 'Reliability_rank', 'Total_Rank']])

threshold_results_mobile = evaluate_thresholds(df_mobile, thresholds_mobile)
print(threshold_results_mobile[['Operator', 'Speed_pass', 'Price_pass', 'Reliability_pass']])