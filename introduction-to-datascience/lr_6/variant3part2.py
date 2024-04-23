import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pandas_datareader import wb
import requests
import json

API_KEY = 'your_api_key_here'
CITY_NAME = 'your_city_name_here'

response = requests.get(f'http://api.openweathermap.org/data/2.5/weather?q={CITY_NAME}&appid={API_KEY}')
data = json.loads(response.text)

pressure = data['main']['pressure']
# Step 1: Fetch data

# Step 2: Statistical forecasting
model = ARIMA(data['pressure'], order=(5,1,0))
model_fit = model.fit(disp=0)
forecast_statistical = model_fit.forecast(steps=3)[0]

# Step 3: Neural network forecasting
X = np.array(range(len(data))).reshape(-1, 1)
y = data['pressure']
model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
model.fit(X, y)
forecast_neural = model.predict(np.array([len(data), len(data)+1, len(data)+2]).reshape(-1, 1))

# Step 4: Compare and display
plt.figure(figsize=(10, 6))
plt.plot(data['pressure'], label='Actual data')
plt.plot(range(len(data), len(data)+3), forecast_statistical, label='Statistical forecast')
plt.plot(range(len(data), len(data)+3), forecast_neural, label='Neural network forecast')
plt.legend()
plt.show()

print('Statistical forecast mean squared error: ', mean_squared_error(data['pressure'][-3:], forecast_statistical))
print('Neural network forecast mean squared error: ', mean_squared_error(data['pressure'][-3:], forecast_neural))