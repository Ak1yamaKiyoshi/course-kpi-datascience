#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Generate the dataset
x = np.linspace(0, 4 * np.pi, 8000)
y = np.cos(x)
noise = np.random.normal(0, 24, y.shape)
y += noise
anomalies = np.random.uniform(np.min(y), np.max(y), int(0.15 * len(y)))
y[:len(anomalies)] = anomalies
data = pd.DataFrame({'x': x, 'y': y})

X_train, X_test, y_train, y_test = train_test_split(data[['x']], data['y'], test_size=0.2, random_state=42)

epochs = [100, 200, 300, 400, 500]
for epoch in epochs:
    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=epoch)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'o', label='Training data', markersize=2)
    plt.plot(X_test, y_test, 'o', label='Test data', markersize=2)
    plt.plot(X_train, y_pred_train, 'o', label='Predicted training data', markersize=2)
    plt.plot(X_test, y_pred_test, 'o', label='Predicted test data', markersize=2)
    plt.legend()
    plt.title(f'Epochs: {epoch}')
    plt.show()

    print(f'Epochs: {epoch}, Training set mean squared error: ', mean_squared_error(y_train, y_pred_train))
    print(f'Epochs: {epoch}, Test set mean squared error: ', mean_squared_error(y_test, y_pred_test))