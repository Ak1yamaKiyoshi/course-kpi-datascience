import pandas as pd
import numpy as np

# Parse the CSV file
df = pd.read_csv("Data_Set_6.csv")


# Clean the data
df = df.replace(['n.a.', 'not avilable'], np.nan)
for month in ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']:
    df[month] = df[month].astype(str).str.replace(r'\D', '', regex=True).replace('', np.nan).astype(float)

# Perform exploratory data analysis
df.info()
df.describe()
df.head()

# Analyze the sales data
total_sales_by_region = df.groupby('SALES_BY_REGION')[['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']].sum()
print(total_sales_by_region)

import matplotlib.pyplot as plt

# Analyze the sales data
total_sales_by_region = df.groupby('SALES_BY_REGION')[['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']].sum()

# Sum up the sales for each month for each region
total_sales_by_region['TOTAL_SALES'] = total_sales_by_region.sum(axis=1)

# Create a bar plot
total_sales_by_region['TOTAL_SALES'].plot(kind='bar')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()