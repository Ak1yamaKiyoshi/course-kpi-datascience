import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def exp_d(size=100000, lambda_val=0.1):
    return np.random.exponential(scale=1.0 / lambda_val, size=size)

def histogram(data, bins=50, label="", xlabel="", ylabel="", title=""):
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='b', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot(data, label="", xlabel="", ylabel="", title=""):
    plt.plot(data, color="gray", alpha=0.6,  marker="x", markersize=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def linear_law(size=1000, slope=0.1, intercept=2, show=False):
    data = np.arange(size) * slope + intercept
    if show:
        plt.plot(data, label='Модель з лінійним законом зміни')
        plt.xlabel('Час/Параметр')
        plt.ylabel('Значення')
        plt.title('Модель з лінійним законом зміни')
        plt.legend()
        plt.show()
    return data 

def constantionous_law(size=1000, show=False):
    data = np.arange(size) * 0.01
    if show: 
        plt.plot(data, label='Модель з постійним законом зміни')
        plt.xlabel('Час/Параметр')
        plt.ylabel('Значення')
        plt.title('Модель з постійним законом зміни')
        plt.legend()
        plt.show()
    return data

def additive_model(data, show=False):
    lambda_val = 0.1
    size = len(data)
    stochastic_component = exp_d(size, lambda_val)
    deterministic_component = data 
    experimental_data = stochastic_component + deterministic_component
    if show: 
        plt.plot(experimental_data, label='Адитивна модель', alpha=0.5)
        plt.plot(stochastic_component, label='Стохастична складова', alpha=0.5)
        plt.plot(deterministic_component, label='Невипадкова складова', alpha=0.5)
        plt.xlabel('Час/Параметр')
        plt.ylabel('Значення')
        plt.title('Адитивна модель експериментальних даних')
        plt.legend()
        plt.show()
    return experimental_data

def monte_carlo_method(data, num_samples=10000):
    return np.random.choice(data, size=num_samples, replace=True)

def variance(data):
    return np.var(data)

def mean(data):
    return np.mean(data)

def deviation(data):
    return (data - np.mean(data)) 

def mean_squared_deviations(data):
    return np.sqrt(np.mean(deviation(data) ** 2))

def overall_analysis(data, label="", xlabel="", ylabel="", title=""):
    print(f" Експерементальні дані \n"
        + f" > Математичне сподівання: {mean(data)}\n"
        + f" > Дисперсія: {variance(data)}\n"
        + f" > Середньоквадратичне відхилення: {mean_squared_deviations(data)}\n"
    )
    histogram(data, label=label, xlabel=xlabel, ylabel=ylabel, title=title)

def normal_d(size=1000, mean=0, std_dev=1):
    return np.random.normal(loc=mean, scale=std_dev, size=size)

if __name__ == '__main__':
    size, lambda_ = 10000, 2
    exponentinal_distribution_model = exp_d(size, lambda_)
    histogram(exponentinal_distribution_model, 50, 'Закон зміни похибки - експоненційний', 'Значення випадкової величини', 'Ймовірність')
    plot(exponentinal_distribution_model, label='Модель зміни досліджуваного процесу експоненційного розподілу', xlabel='Час', ylabel='Значення')
    additive_model_ = additive_model(constantionous_law(size, show=True), show=True)
    overall_analysis(monte_carlo_method(additive_model_, len(additive_model_)//4), 'Розподіл адитивної моделі з методом Монте-Карло', 'Значення', 'Ймовірність', 'Розподіл адитивної моделі')
    overall_analysis(additive_model_, 'Адитивна модель', 'Значення', 'Ймовірність', 'Адитивна модель')