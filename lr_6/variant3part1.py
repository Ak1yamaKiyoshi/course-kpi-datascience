import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x = np.linspace(0, 4 * np.pi, 6000)
y = np.sin(x)
noise = np.random.normal(0, 15, y.shape)
y += noise
anomalies = np.random.uniform(np.min(y), np.max(y), int(0.6 * len(y)))
y[:len(anomalies)] = anomalies
data = pd.DataFrame({'x': x, 'y': y})

X_train, X_test, y_train, y_test = train_test_split(data[['x']], data['y'], test_size=0.2, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
model.fit(X_train, y_train)


X_train, X_test, y_train, y_test = train_test_split(data[['x']], data['y'], test_size=0.2, random_state=42)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

network_structures = [
    (50,),  
    (100, 50),    
    (50, 50, 50), 
    (50, 50, 100),
]


for structure in network_structures:
    model = MLPRegressor(hidden_layer_sizes=structure, activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'o', label='Training data', markersize=2)
    plt.plot(X_test, y_test, 'o', label='Test data', markersize=2)
    plt.plot(X_train, y_pred_train, 'o', label=f'Predicted training data ({structure})', markersize=2)
    plt.plot(X_test, y_pred_test, 'o', label=f'Predicted test data ({structure})', markersize=2)
    plt.legend()
    plt.show()

    print(f'Training set mean squared error ({structure}): ', mean_squared_error(y_train, y_pred_train))
    print(f'Test set mean squared error ({structure}): ', mean_squared_error(y_test, y_pred_test))