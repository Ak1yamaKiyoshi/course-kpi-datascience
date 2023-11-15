import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math as mt

def parse_to_numpy(filename: str, column: str, delimiter: str = ",") -> np.ndarray:
    """ Takes given column from csv file with filename and returns it as numpy array """
    if (column not in pd.read_csv(filename, delimiter=delimiter, dtype=str).columns):
        raise Exception(f"Column '{column}' not found in {filename}; \n columns: {pd.read_csv(filename, delimiter=delimiter, dtype=str).columns}")    
    
    return pd.read_csv(filename, delimiter=delimiter)[column].to_numpy()


def parse_date_to_datetime(date: str) -> dt.datetime:
    return dt.datetime.strptime(date, '%m/%d/%Y').date()


def week_difference_in_dates(date1: str, date2: str) -> int:
    return (parse_date_to_datetime(date1) - parse_date_to_datetime(date2)).days//7


def mat_plot(data: np.ndarray, 
             values_to_mark: [int, int] = None,
             values_to_mark_labels: [str] = None,
             marker_values_to_mark: str = "v",
             values_to_mark_color: str = "red",
             marker_size: int = 10, 
             data_label="",
             display_legend: bool=False,
             display_grid: bool=False, 
             color: str = "cyan",
             xlim: [int, int] = None,
             ylim: [int, int] = None,
             xlabel: str = "", 
             ylabel: str = "",
             disable_toolbox: bool = True,
             figure_label: str = "",
             figure_label_fontsize: int = 14,
             figure_label_fontweight: str = "bold" 
             ) -> None: 
    
    if disable_toolbox: plt.rcParams['toolbar'] = 'None'
    plt.suptitle(figure_label, fontsize=figure_label_fontsize, fontweight=figure_label_fontweight)
    plt.plot(data, label=data_label, color=color)
    if values_to_mark: 
        for i in range(len(values_to_mark)): 
            plt.plot(values_to_mark[i][0], values_to_mark[i][1], marker=marker_values_to_mark, 
                     markersize=marker_size, color=values_to_mark_color, 
                     label=values_to_mark_labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    if display_grid: plt.grid()
    if display_legend: plt.legend()
    plt.show()
    

def model_ideal(n): 
    model = np.zeros((n)) # Quadratic reciprocity law 
    for i in range(n): 
        model[i]=(0.0000005*i*i) 
    return model
    
def uniform_abnormal_measurments(n, nav, to_print: bool = False) -> None:
    SAV = np.random.randint(n, size=nav) #np.zeros((nav))
    if to_print: 
        S = np.random.randint(n*10, size=n) # 
        mS = np.median(S)                   # Математичне сподівання 
        dS = np.var(S)                      # Дисперсія 
        scvS = mt.sqrt(dS)                  # Відхилення вибірки випадкової величини від середньоквадратичної функції 
        print(f" > AB numbers: {SAV} \n > mathematical expectation {mS} \n > variance {dS} \n > standard deviation {scvS}   ")
    return SAV


def errors_normal(SN, noise, samples, to_print: bool = False):
    S = np.random.normal(SN, noise, samples)
    if to_print: 
        mS = np.median(S)
        dS = np.var(S)
        scvS = mt.sqrt(dS)
        print(f" <normal noise> \n > mathematical expectation {mS} \n > variance {dS} \n > standard deviation {scvS}   ")
    return S


def Model_NORM (SN, S0N, n):
    SV=np.zeros((n))
    for i in range(n):
        SV[i] = S0N[i]+SN[i]
    return SV

def least_squares(data): 
    squares_sum = np.sum(data*data)
    
def main(): 
    # > Choose dataset to work with
    datasets = { "wsl": "EUR_RUB_wsj.csv",
         "tradingview": "EUR_RUB_trading_view.csv" }
    CURRENT_DATASET = datasets["wsl"]
    
    # > Parse needed data from dataset 
    data = {"date": np.flip(parse_to_numpy(CURRENT_DATASET, "Date", delimiter=","), 0), 
           "price": np.flip(parse_to_numpy(CURRENT_DATASET, "Price", delimiter=","), 0)}

    # > Define Constants    
    n    = 1000   # random value sample volume
    q_av = 3      # abnormal measurements coefficient | коефіцієнт переваги АВ - Аномальні виміри 
    navv = 10   
    nav  = (n*navv)//100 # amount of abnormal measurements | кількість АВ у відсотках та абсолютних одиницях
    dm   = 0,     #
    dsig = 5      #  параметри нормального закону розподілу ВВ: середнє та середньоквадратиче відхилення
    
    # > Define models
    model_i = model_ideal(n) # модель ідеального тренду (квадратичний закон)
    model_a = uniform_abnormal_measurments(n, nav)  # модель рівномірних номерів АВ
    model_e = errors_normal(dm, dsig, n)       # модель нормальних помилок
    
    
    # > Errors 
    #SV = Model_NORM(S, S0, n)  # модель тренда + нормальних помилок
    #Plot_AV(S0, SV, 'квадратична модель + Норм. шум')
    #Stat_characteristics(SV, 'Вибірка + Норм. шум')
    
    # > plot with EUR / RUB exchange rate data from dataset   
    mat_plot(data["price"], data_label="eur/rub", display_grid=True, 
             display_legend=True, values_to_mark_color="red", marker_values_to_mark=">",
             values_to_mark=[
                [np.where(data["price"] == (dataset_price_max := np.amax(data["price"]))), dataset_price_max],
                [np.where(data["price"] == (dataset_price_min := np.amin(data["price"]))), dataset_price_min]],
             values_to_mark_labels=["highest price", "lowest price"], color="blue",
             xlim=[0, len(data["price"])-1], ylim=[dataset_price_min - 10, dataset_price_max + 10],
             xlabel=f"Weeks {data['date'][0]} - {data['date'][-1]}",  ylabel="Price",
             figure_label=f"EUR/RUB price in period {data['date'][0]} - {data['date'][-1]}"
    )
    
    # > plot with ideal model
    mat_plot(model_i, color="blue", figure_label="Ideal model")
    # > plot with abnormal measurments
    mat_plot(model_a, color="blue", figure_label="Abnormal measurments")
    # > plot with normal errors(noise)
    mat_plot(model_e, color="blue", figure_label="Normal errors(noise)")

if __name__ == "__main__": 
    main()


