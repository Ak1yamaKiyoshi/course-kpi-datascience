# 7, 22 Розробити програмний скрипт, що реалізує аналіз даних, поданих у файлі Data_Set_7.csv

import matplotlib.pyplot as plt



import pandas as pd
import numpy as np

# Parse the CSV file
df = pd.read_csv("Data_Set_6.csv")


df = df.replace(['n.a.', 'not avilable'], np.nan)
for month in ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']:
    df[month] = df[month].astype(str).str.replace(r'\D', '', regex=True).replace('', np.nan).astype(float)


total_sales_by_region = df.groupby('SALES_BY_REGION')[['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']].sum()
print(total_sales_by_region)

total_sales_by_region = df.groupby('SALES_BY_REGION')[['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']].sum()

total_sales_by_region['TOTAL_SALES'] = total_sales_by_region.sum(axis=1)

total_sales_by_region['TOTAL_SALES'].plot(kind='bar')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()