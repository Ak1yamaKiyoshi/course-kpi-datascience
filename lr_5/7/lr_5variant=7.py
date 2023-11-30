import pandas as pd
import numpy as np


np.random.seed(42)
data = {
    'Назва товару': ['Комплекс1', 'Комплекс2', 'Комплекс3', 'Комплекс4', 'Комплекс5'],
    'Формат файлу': ['CSV', 'XML', 'JSON', 'CSV', 'XML'],
    'Тип товару': ['Сервер', 'Робоча станція', 'Хмарний сервіс', 'Сервер', 'Хмарний сервіс'],
    'Кількість аналогічних товарів': [20, 18, 25, 22, 20],
    'Швидкодія': [8, 9, 7, 8, 9],
    'Надійність': [9, 8, 7, 9, 8],
    'Цінова категорія': ['Висока', 'Середня', 'Висока', 'Висока', 'Середня'],
    'Можливості розширення': [7, 8, 9, 7, 8],
    'Технічна підтримка': [8, 9, 7, 8, 9],
    'Енергоефективність': [9, 8, 7, 9, 8],
    'Масштабованість': [8, 9, 7, 8, 9],
    'Витрати на обслуговування': [7, 8, 9, 7, 8],
    'Ефективність використання ресурсів': [9, 8, 7, 9, 8],
    'Сумісність': [8, 9, 7, 8, 9],
    'Інтеграція з іншими системами': [9, 8, 7, 9, 8],
    'Вартість': [3000, 2500, 3500, 2800, 3200],
}

df = pd.DataFrame(data)
df.to_csv('computing_complex_data.csv', index=False)


minimize_criteria = ['Витрати на обслуговування', 'Цінова категорія', 'Вартість']
maximize_criteria = ['Швидкодія', 'Надійність', 'Можливості розширення', 'Технічна підтримка', 'Енергоефективність']

df_minimized = df.copy()
df_minimized['Цінова категорія'] = pd.to_numeric(df_minimized['Цінова категорія'], errors='coerce')
df_minimized[minimize_criteria] = df_minimized[minimize_criteria].apply(lambda x: 1 / x)

df_maximized = df.copy()
df_maximized[maximize_criteria] = df_maximized[maximize_criteria].apply(lambda x: x / x.max())


df_minimized.to_csv('minimized_computing_complex_data.csv', index=False)
df_maximized.to_csv('maximized_computing_complex_data.csv', index=False)


mean_minimized = df_minimized.describe().transpose()['mean']
mean_maximized = df_maximized.describe().transpose()['mean']

analysis_df = pd.DataFrame({
    'Mean Minimized': mean_minimized[minimize_criteria],
    'Mean Maximized': mean_maximized[maximize_criteria],
})

print(analysis_df)