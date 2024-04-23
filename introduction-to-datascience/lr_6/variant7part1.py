import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Generate the dataset
x = np.linspace(0, 10, 5000)
y = x
noise = np.random.normal(0, 20, y.shape)
y += noise
anomalies = np.random.uniform(np.min(y), np.max(y), int(0.1 * len(y)))
y[:len(anomalies)] = anomalies
data = pd.DataFrame({'x': x, 'y': y})

# Step 2: Create the neural network
X_train, X_test, y_train, y_test = train_test_split(data[['x']], data['y'], test_size=0.2, random_state=42)

# Step 3: Display the prediction process and Step 4: Investigate the dependence of the accuracy
layers = [(50,), (100,), (200,), (300,), (400,), (500,)]
for layer in layers:
    model = MLPRegressor(hidden_layer_sizes=layer, activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'o', label='Training data', markersize=2)
    plt.plot(X_test, y_test, 'o', label='Test data', markersize=2)
    plt.plot(X_train, y_pred_train, 'o', label='Predicted training data', markersize=2)
    plt.plot(X_test, y_pred_test, 'o', label='Predicted test data', markersize=2)
    plt.legend()
    plt.title(f'Hidden Layer Sizes: {layer}')
    plt.show()

    print(f'Hidden Layer Sizes: {layer}, Training set mean squared error: ', mean_squared_error(y_train, y_pred_train))
    print(f'Hidden Layer Sizes: {layer}, Test set mean squared error: ', mean_squared_error(y_test, y_pred_test))