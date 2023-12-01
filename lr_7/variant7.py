import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Pr12_2.csv")
df.info()
df.describe()
df.head()

df['Number of sales'] = pd.to_numeric(df['Number of sales'], errors='coerce')

df = df.dropna(subset=['Number of sales'])

# Calculate profit
df['Profit'] = df['Number of sales'] * (df['Selling price'] - df['Unit cost'])

df = df.dropna()


df['Number of sales'] = pd.to_numeric(df['Number of sales'], errors='coerce')

df['Profit'] = df['Number of sales'] * (df['Selling price'] - df['Unit cost'])

model = LinearRegression()
model.fit(df['Number of sales'].values.reshape(-1,1), df['Profit'])

predictions = model.predict(df['Number of sales'].values.reshape(-1,1))

plt.scatter(df['Number of sales'], df['Profit'])
plt.plot(df['Number of sales'], predictions, color='red')
plt.show()

df['Number of sales'] = pd.to_numeric(df['Number of sales'], errors='coerce')

df['Profit'] = df['Number of sales'] * (df['Selling price'] - df['Unit cost'])

df['Profit'].plot(kind='line')
plt.title('Profit over time')
plt.xlabel('Index')
plt.ylabel('Profit')
plt.show()

df['Profit'].hist(bins=30)
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.show()