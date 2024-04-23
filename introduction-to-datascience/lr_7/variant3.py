import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Pr12_2.csv")
df.info()
df.describe()
df.head()

# Convert 'Number of sales' to numeric
df['Number of sales'] = pd.to_numeric(df['Number of sales'], errors='coerce')

# Drop rows with NaN values in 'Number of sales'
df = df.dropna(subset=['Number of sales'])

# Calculate profit
df['Profit'] = df['Number of sales'] * (df['Selling price'] - df['Unit cost'])

#Drop rows with NaN values 
df = df.dropna()

# Convert 'Number of sales' to numeric
df['Number of sales'] = pd.to_numeric(df['Number of sales'], errors='coerce')

# Calculate profit
df['Profit'] = df['Number of sales'] * (df['Selling price'] - df['Unit cost'])

# Define the mathematical model of the data according to the least squares method (OLS)
model = LinearRegression()
model.fit(df['Number of sales'].values.reshape(-1,1), df['Profit'])

# Make a prediction of the dynamics of profit changes
predictions = model.predict(df['Number of sales'].values.reshape(-1,1))

# Plot the results
plt.scatter(df['Number of sales'], df['Profit'])
plt.plot(df['Number of sales'], predictions, color='red')
plt.show()

# Convert 'Number of sales' to numeric
df['Number of sales'] = pd.to_numeric(df['Number of sales'], errors='coerce')

# Calculate profit
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