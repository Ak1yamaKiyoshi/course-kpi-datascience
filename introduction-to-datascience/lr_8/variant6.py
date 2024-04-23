
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv')

# Strip leading and trailing whitespace
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Replace 'n.a.' and 'not avilable' with NaN
df = df.replace(['n.a.', 'not avilable', 'not available'], np.nan)

# Remove commas from numeric columns
df.iloc[:, 2:] = df.iloc[:, 2:].replace({',': ''}, regex=True)

# Convert numeric columns to numeric type
df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

df = df.dropna()
df = df.reset_index(drop=True)

print(df.info())
print(df.describe())
print(df.head(5))



# Monthly sales for each month
months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
for i, month in enumerate(months):
    df[f'sales_month_{i+1}'] = df[month]


months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
for i, month in enumerate(months):
    df[f'sales_month_{i+1}'] = df[month]

def calculate_integrated_score(df):
    # This will depend on your specific scoring model
    df['Scor'] = df.iloc[:, 2:].mean(axis=1)  
    return df

df = calculate_integrated_score(df)


kmeans = KMeans(n_clusters=2)
df['scor_high'] = (df['Scor'] > df['Scor'].median()).astype(int)
df['cluster'] = kmeans.fit_predict(df[['Scor', 'scor_high']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Scor', y='scor_high', hue='cluster')
plt.title('Clustering Results')
plt.show()

df['SALES_BY_REGION'] = pd.factorize(df['SALES_BY_REGION'])[0]
print(df.columns)

sns.heatmap(df.corr(), annot=False)
plt.show()