import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data_Set_3.csv")

print(df.info())
print(df.describe())
print(df.head())

# Convert 'Unit Cost' and 'Total' to numeric
df['Unit Cost'] = df['Unit Cost'].str.replace(r'[^\d.]', '', regex=True).astype(float)
df['Total'] = df['Total'].str.replace(r'[^\d.]', '', regex=True).astype(float)


print("\nsales by region ")
total_sales_by_region = df.groupby('Region')['Total'].sum()
print(total_sales_by_region)

# Calculate total sales for each product
print("\nsales by product ")
total_sales_by_product = df.groupby('Item')['Total'].sum()
print(total_sales_by_product)

df['Total'].plot(kind='line')
plt.title('Sales over time')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.show()

df['Total'].hist(bins=30)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()