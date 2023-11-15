import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math as mt
import scipy as sci




# Закон зміни похибки – експонентційний;
def exp_d(size=100000, lambda_val=0.1):
    """ Exponential Distribution array generator """
    return np.random.exponential(scale=1.0 / lambda_val, size=size)
   
def histogram(data, bins=50, label="", xlabel="", ylabel="", title=""):
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='b', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # Show the plot
    plt.show()
   
def plot(data, label="", xlabel="", ylabel="", title=""):
    plt.plot(data, color="gray", alpha=0.6,  marker="x", markersize=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()  


def linear_law(size=1000, slope=0.1, intercept=2, show=False):
    # Створення моделі з лінійним законом зміни
    def linear_change_model(size, slope=0.1, intercept=0):
        time = np.arange(size)  # Часова відмітка
        data = slope * time + intercept  # Лінійна залежність
        return data


    data = linear_change_model(size, slope, intercept)
    if show:
        # Візуалізація результатів
        plt.plot(data, label='Модель з лінійним законом зміни')
        plt.xlabel('Час/Параметр')
        plt.ylabel('Значення')
        plt.title('Модель з лінійним законом зміни')
        plt.legend()
        plt.show()
    return data


def additive_model(data, show=False):
    # Параметри експоненційного розподілу
    lambda_val = 0.1  # Середній інтервал між подіями (1/λ)
    size = len(data)  # Розмір вибірки
    # Генерація стохастичної та невипадкової складових
    stochastic_component = exp_d( size, lambda_val)
    # data - невипадкова складова
    deterministic_component = data
    # Сумування стохастичної та невипадкової складових для адитивної моделі
    experimental_data = stochastic_component + deterministic_component
    if show:
        # Візуалізація результатів
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
    """ Метод монте карло ( випадкова вибірка з даних ) """
    return np.random.choice(data, size=num_samples, replace=True)


def variance(data):
    """ Дисперсія """
    return np.var(data)


def mean(data):
    """ Математичне сподівання """
    return np.mean(data)


def deviation(data):
    """ Відхилення """
    return (data - np.mean(data))


def mean_squared_deviations(data):
    """ Середньоквадратичне відхилення """
    # 1. Обчислення квадратів відхилень від середнього значення
    # 2. Обчислення середнього значення квадратів відхилень
    # 3. Корінь з середнього значення квадратів відхилень
    return np.sqrt(np.mean(deviation(data) ** 2))




def overall_analysis(data, label="", xlabel="", ylabel="", title=""):
    print(f" Експерементальні дані \n"
        + f" > Математичне сподівання: {mean(data)}\n"
        + f" > Дисперсія: {variance(data)}\n"
        + f" > Середньоквадратичне відхилення: {mean_squared_deviations(data)}\n"
    )
    histogram(data, label=label, xlabel=xlabel, ylabel=ylabel, title=title)


def normal_d(size=1000, mean=0, std_dev=1):
    """ Генератор нормального розподілу """
    return np.random.normal(loc=mean, scale=std_dev, size=size)


if __name__ == '__main__':
   
    """  -(3) Закон зміни похибки – експонентційний;
      Закон зміни досліджуваного процесу – постійна. """
     
    size, lambda_ = 10000, 2
    exponentinal_distribution_model = exp_d(size, lambda_) #normal_d(size, mean=0, std_dev=0.1) #
    """ 1. Модель генерації випадкової величини – похибки вимірювання за заданим у таблиці Д1 додатку 1 закону розподілу """
    histogram(exponentinal_distribution_model, 50, 'Закон зміни похибки - експоненційний', 'Значення випадкової величини', 'Ймовірність')
    """ 2. Модель зміни досліджуваного процесу за заданим у таблиці Д1 додатку 1 закону """
    plot(exponentinal_distribution_model, label='Модель зміни досліджуваного процесу експоненційного розподілу', xlabel='Час', ylabel='Значення')
    """ 3.  Адитивну модель експериментальних даних (вимірів досліджуваного процесу) відповідно до синтезованих в п.1,2 моделей випадкової (стохастична) і невипадкової складових. """
    additive_model_ = additive_model(linear_law(size, show=True), show=True)
    #additive_model_ = additive_model(constantionous_law(size), show=True)
    """ 4. Метод Монте-Карло для дослідження статистичних характеристик експериментальних даних"""
    overall_analysis(monte_carlo_method(additive_model_, len(additive_model_)//4), 'Розподіл адитивної моделі з методом Монте-Карло', 'Значення', 'Ймовірність', 'Розподіл адитивної моделі')
    """ 5. Визначення статистичних (числових) характеристик експериментальних даних (дисперсію, середньоквадратичне відхилення математичне сподівання, гістограми закону розподілу похибки та експериментальних даних)."""
    overall_analysis(additive_model_, 'Адитивна модель', 'Значення', 'Ймовірність', 'Адитивна модель')