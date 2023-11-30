import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_dataset():
    # Згенеруємо приклади для датасету
    np.random.seed(42)
    data = {
        'Назва товару': ['Товар1', 'Товар2', 'Товар3', 'Товар4', 'Товар5'],
        'Категорія товару': ['Електроніка', 'Одяг', 'Харчові продукти', 'Електроніка', 'Одяг'],
        'Цінова категорія': ['Середня', 'Висока', 'Низька', 'Середня', 'Висока'],
        'Цінова стратегія': ['Знижки', 'Пакетні пропозиції', 'Знижки', 'Знижки', 'Пакетні пропозиції'],
        'Цільова аудиторія': ['Молодь', 'Дорослі', 'Сім\'я', 'Молодь', 'Дорослі'],
        'Конкуренти': ['Конкурент1', 'Конкурент2', 'Конкурент3', 'Конкурент1', 'Конкурент2'],
        'Маркетингові стратегії': ['Реклама', 'Соціальні мережі', 'Реклама', 'Реклама', 'Соціальні мережі'],
        'Потенційний обсяг ринку': [1000, 800, 1200, 900, 1000],
        'Інноваційні рішення': ['Технологія A', 'Дизайн B', 'Технологія A', 'Технологія C', 'Дизайн B'],
        'Ступінь конкурентоспроможності': [4, 3, 5, 4, 3],
        'Прогноз продажів': [500, 400, 600, 450, 500],
        'Відгуки та рейтинги': [4.5, 4.0, 4.8, 4.2, 4.1]
    }

    df = pd.DataFrame(data)
    df.to_csv('variant6.csv', index=False)

create_dataset()

df = pd.read_csv('variant6.csv')

minimize_criteria = ['Цінова категорія', 'Потенційний обсяг ринку', 'Ступінь конкурентоспроможності']
maximize_criteria = ['Прогноз продажів', 'Відгуки та рейтинги']

# Мінімізуємо параметри
df_minimized = df.copy()
df_minimized['Цінова категорія'] = pd.to_numeric(df_minimized['Цінова категорія'], errors='coerce')
df_minimized[minimize_criteria] = df_minimized[minimize_criteria].apply(lambda x: 1 / x)

# Максимізуємо параметри
df_maximized = df.copy()
df_maximized[maximize_criteria] = df_maximized[maximize_criteria].apply(lambda x: x / x.max())

# Зберігаємо результат у нові файли
df_minimized.to_csv('minimized_data.csv', index=False)
df_maximized.to_csv('maximized_data.csv', index=False)




df_minimized.describe().transpose()[['mean', 'std']].plot(kind='bar', y='mean', yerr='std', legend=False)
plt.title('Середнє значення та стандартне відхилення для мінімізованих параметрів')
plt.show()

df_maximized.describe().transpose()[['mean', 'std']].plot(kind='bar', y='mean', yerr='std', legend=False)
plt.title('Середнє значення та стандартне відхилення для максимізованих параметрів')
plt.show()

mean_minimized = df_minimized.describe().transpose()['mean']
mean_maximized = df_maximized.describe().transpose()['mean']

analysis_df = pd.DataFrame({
    'Mean Minimized': mean_minimized[minimize_criteria],
    'Mean Maximized': mean_maximized[maximize_criteria],
})

print(analysis_df)