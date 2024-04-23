import pandas as pd




def normalization(data):
    data['normalized_time'] = 1 - (data['time'] - data['time'].min()) / (data['time'].max() - data['time'].min())
    data['normalized_distance'] = 1 - (data['distance'].str.extract('(\d+)').astype(int) - data['distance'].str.extract('(\d+)').astype(int).min()) / (data['distance'].str.extract('(\d+)').astype(int).max() - data['distance'].str.extract('(\d+)').astype(int).min())
    data['normalized_price'] = 1 - (data['price'] - data['price'].min()) / (data['price'].max() - data['price'].min())

    data['total_score'] = data['normalized_time'] + data['normalized_distance'] + data['normalized_price']
    return data


def ranking(data):
    weights = {'time': 0.4, 'distance': 0.3, 'price': 0.3}

    data['total_rank'] = (data['time'] * weights['time'] +
                        data['distance'].str.extract('(\d+)', expand=False).astype(int) * weights['distance'] +
                        data['price'] * weights['price'])
    return data

def electre(data):
    thresholds = {'time': 30, 'distance': 5, 'price': 150}
    data['distance_to_threshold_time'] = abs(data['time'] - thresholds['time'])
    data['distance_to_threshold_distance'] = abs(data['distance'].str.extract('(\d+)').astype(int) - thresholds['distance'])
    data['distance_to_threshold_price'] = abs(data['price'] - thresholds['price'])

    relation_matrix = pd.DataFrame(index=data.index, columns=data.index)

    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                relation_matrix.iloc[i, j] = (
                    (data['distance_to_threshold_time'].iloc[i] >= data['distance_to_threshold_time'].iloc[j]) and
                    (data['distance_to_threshold_distance'].iloc[i] >= data['distance_to_threshold_distance'].iloc[j]) and
                    (data['distance_to_threshold_price'].iloc[i] >= data['distance_to_threshold_price'].iloc[j])
                )

    data['importance'] = relation_matrix.sum(axis=1)
    return data

data = pd.read_csv('varaint_3.csv')
print(normalization(data))
print(ranking(data))
print(electre(data))
print(data[['time', 'distance', 'price', 'total_score', 'total_rank', 'importance']])