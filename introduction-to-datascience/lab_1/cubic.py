import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math as mt
from scipy.signal import savgol_filter
import scipy as sci

def parse_to_numpy(filename: str, column: str, delimiter: str = ",") -> np.ndarray:
    """ Takes given column from csv file with filename and returns it as numpy array """
    if (column not in pd.read_csv(filename, delimiter=delimiter, dtype=str).columns):
        raise Exception(f"Column '{column}' not found in {filename}; \n columns: {pd.read_csv(filename, delimiter=delimiter, dtype=str).columns}")    
    
    return pd.read_csv(filename, delimiter=delimiter)[column].to_numpy()
    
# > Choose dataset to work with
datasets = { "wsl": "EUR_RUB_wsj.csv",
        "tradingview": "EUR_RUB_trading_view.csv" }
CURRENT_DATASET = datasets["wsl"]

# > Parse needed data from dataset 
data = {"date": np.flip(parse_to_numpy(CURRENT_DATASET, "Date", delimiter=","), 0), 
        "price": np.flip(parse_to_numpy(CURRENT_DATASET, "Price", delimiter=","), 0),
        "indicies": np.fromiter(range(len(parse_to_numpy(CURRENT_DATASET, "Date", delimiter=","))),dtype=int)
}  

def mean_squarred_error(a: np.ndarray, b: np.ndarray) -> float:
    return ((a - b)**2).mean()

def get_best_polyfit(data_x: np.ndarray, data_y: np.ndarray) -> np.polyfit:
    polyfit = np.polyfit(data_x, data_y, 90)
    poly1d = np.poly1d(polyfit)
    
    best_mse = None 
    best_polyfit = None
    best_degree = None
    for i in range(0, 90): 
        polyfit = np.polyfit(data_x, data_y, i)
        poly1d = np.poly1d(polyfit)
        
        y_model = poly1d(np.arange(0, len(data_x), 1))
        mse = (( data_y - y_model)**2).mean()
        if best_mse is None or mse < best_mse: 
            best_mse = mse
            best_polyfit = polyfit
            best_degree = i
    return {
        "polyfit": best_polyfit,
        "degree": best_degree,
        "mse": best_mse    
    }

# > Trend 
best_poly_data = get_best_polyfit(data["indicies"], data["price"])
polyfit = best_poly_data["polyfit"]
poly1d  = np.poly1d(polyfit)
y_model = poly1d(np.arange(0, len(data["indicies"]), 1))

plt.plot(data["price"], color="gray", label="Eur/Rub exchange rate")
plt.plot(y_model, "--", color="blue", label="Trend, polynomial")
#plt.show()

yhat = savgol_filter(data["price"], 51, 3) # window size 51, polynomial order 3
plt.plot(yhat, "--", color="red", label="Trend, savgov filter")

print("mse savgov filter: ",mean_squarred_error(data["price"], yhat))
print("mse polyfit: ", best_poly_data["mse"])

plt.legend()
plt.show()
print(f"Best degree: {best_poly_data['degree']}")
