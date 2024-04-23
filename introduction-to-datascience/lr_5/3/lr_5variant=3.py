import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


np.random.seed(42)
num_samples = 8
num_criteria = 12

criteria_names = ['Ціна', 'Рік побудови', 'Років з останнього ремонту', 'Кількість кімнат',
                'Тип житла', 'Відстань від центру', 'Наявність гостьової кімнати',
                'Тип опалення', 'Вартість оренди', 'Інший критерій1', 'Інший критерій2', 'Інший критерій3']

data = np.random.randint(1, 100, size=(num_samples, num_criteria))


df = pd.DataFrame(data, columns=criteria_names)
df.to_csv('variant3.csv', index=False)

weights = np.random.rand(num_criteria)
weights /= weights.sum()

print("Важливість критеріїв:")
for i, name in enumerate(criteria_names):
    print(f"{name}: {weights[i]:.3f}")


minimize_columns = ['Років з останнього ремонту', 'Відстань від центру', 'Вартість оренди', 'Інший критерій1']
scaler = MinMaxScaler()
df[minimize_columns] = scaler.fit_transform(df[minimize_columns])


df['Ефективність'] = np.dot(df.values, weights)

df = df.sort_values(by='Ефективність', ascending=False)

print("\nРезультати сортування:")
print(df[['Ефективність'] + criteria_names])

df.to_csv('variant3sorted.csv', index=False)