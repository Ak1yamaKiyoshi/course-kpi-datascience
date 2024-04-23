import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Алгоритм виявлення аномальних вимірів: Метод найменших квадратів
def detect_anomalies(data):
    """
    Виявляє аномальні виміри за допомогою порогового методу.


    Parameters:
    - data: numpy array, вхідні виміри


    Returns:
    - anomalies: numpy array, індекси аномальних вимірів
    """
    threshold = 7.5
    anomalies = np.where(data > threshold)[0]
    return anomalies


def detect_anomalies_decay(data, alpha=0.05):
    """
    Алгоритм виявлення аномалій за коефіцієнтом старіння інформації.


    Parameters:
    - data: numpy array, вхідні виміри
    - alpha: float, коефіцієнт старіння інформації


    Returns:
    - anomalies: numpy array, індекси аномальних вимірів
    """
    anomalies = []
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) > alpha:
            anomalies.append(i)
    return np.array(anomalies)




# Метод усунення впливу аномальних вимірів: Відновлення вимірів методом найменших квадратів
def restore_measurements(data, anomalies):
    """
    Метод усунення впливу аномальних вимірів (відновлення).


    Parameters:
    - data: numpy array, вхідні виміри
    - anomalies: numpy array, індекси аномальних вимірів


    Returns:
    - restored_data: numpy array, виміри після відновлення аномалій методом найменших квадратів
    """
    restored_data = np.copy(data)
    restored_data[anomalies] = np.nan  # Позначте аномалії для відновлення
    valid_indices = np.where(~np.isnan(restored_data))[0]
    restored_data[anomalies] = np.polyval(np.polyfit(valid_indices, restored_data[valid_indices], 2), anomalies)
    return restored_data


# Здійснення згладжування методом найменших квадратів
def smooth_data(data, anomalies):
    """
    Згладжує виміри методом найменших квадратів.


    Parameters:
    - data: numpy array, вхідні виміри
    - anomalies: numpy array, індекси аномальних вимірів


    Returns:
    - smoothed_data: numpy array, виміри після згладжування аномалій методом найменших квадратів
    """
    valid_indices = np.where(~np.isnan(data))[0]
    smoothed_data = np.copy(data)
    smoothed_data[anomalies] = np.polyval(np.polyfit(valid_indices, data[valid_indices], 2), anomalies)
    return smoothed_data


# Здійснення Монте-Карло аналізу
def monte_carlo_analysis(data, anomalies):
    """
    Монте-Карло аналіз для оцінки результатів згладжування.


    Parameters:
    - data: numpy array, вхідні виміри
    - anomalies: numpy array, індекси аномальних вимірів


    Returns:
    - results: dictionary, результати Монте-Карло аналізу
    """
    n_iterations = 1000
    errors_noisy = []
    errors_anomalous = []
    errors_smoothed = []


    for _ in range(n_iterations):
        noisy_data = data + np.random.normal(0, 1, len(data))
        errors_noisy.append(np.mean((noisy_data - ideal_values) ** 2))


        anomalous_data = np.copy(noisy_data)
        anomalous_data[anomalies] += 10
        errors_anomalous.append(np.mean((anomalous_data - ideal_values) ** 2))


        smoothed = smooth_data(noisy_data, anomalies)
        errors_smoothed.append(np.mean((smoothed - ideal_values) ** 2))


    return {
        'errors_noisy': errors_noisy,
        'errors_anomalous': errors_anomalous,
        'errors_smoothed': errors_smoothed
    }


# Відображення результатів
def display_results(ideal, noisy, noisy_with_anomalies, smoothed, errors_noisy, errors_anomalous, errors_smoothed):
    """
    Відображає результати експерименту.
    Parameters:
    - ideal: numpy array, ідеальні виміри
    - noisy: numpy array, зашумлені виміри без аномалій
    - noisy_with_anomalies: numpy array, зашумлені виміри з аномаліями
    - smoothed: numpy array, виміри після згладжуван
    - errors_noisy: numpy array, масив квадратичних похибок для зашумлених вимірів без аномалій
    - errors_anomalous: numpy array, масив квадратичних похибок для зашумлених вимірів з аномаліями
    - errors_smoothed: numpy array, масив квадратичних похибок для вимірів після згладжування
    """


    statistics_table = pd.DataFrame({
        'Параметр': ['Математичне сподівання', 'Середньоквадратичне відхилення'],
        'Без аномалій': [np.mean(errors_noisy), np.std(errors_noisy)],
        'З аномаліями': [np.mean(errors_anomalous), np.std(errors_anomalous)],
        'Після згладжування МНК': [np.mean(errors_smoothed), np.std(errors_smoothed)]
    })
    print(statistics_table)


    plt.figure(figsize=(10, 15))


    # Перший графік - Квадратичний тренд
    plt.subplot(4, 1, 1)
    plt.plot(ideal, label='Квадратичний тренд', color='blue')
    plt.legend()
    plt.title('Квадратичний тренд')


    # Другий графік - Зашумлена без аномалій
    plt.subplot(4, 1, 2)
    plt.plot(noisy, label='Зашумлена без аномалій', color='orange')
    plt.legend()
    plt.title('Зашумлена без аномалій')


    # Третій графік - Зашумлена з аномаліями
    #plt.title('Зашумлена з аномаліями')
    plt.subplot(4, 1, 3)
    plt.plot(noisy_with_anomalies, label='Зашумлена з аномаліями', color='green')


    # Перевірка, чи аномалії в межах вимірів
    valid_anomalies = anomalies[anomalies < len(noisy_with_anomalies)]
    plt.scatter(valid_anomalies, noisy_with_anomalies[valid_anomalies], color='red', label='Аномалії')
    plt.legend()
    plt.title('Зашумлена з аномаліями')


    # Четвертий графік - Результат згладжування МНК
    plt.subplot(4, 1, 4)
    plt.plot(smoothed, label='Результат згладжування МНК', linestyle='--', color='purple')
    plt.legend()
    plt.title('Результат згладжування МНК')


    # Додаткові налаштування
    plt.tight_layout()
    plt.show()


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Гістограма для без аномалій
    axs[0].hist(errors_noisy, alpha=0.5, color='blue', label='Без аномалій')
    axs[0].set_title('Без аномалій')


    # Гістограма для з аномаліями
    axs[1].hist(errors_anomalous, alpha=0.5, color='orange', label='З аномаліями')
    axs[1].set_title('З аномаліями')


    # Гістограма після згладжування МНК
    axs[2].hist(errors_smoothed, alpha=0.5, color='green', label='Після згладжування МНК')
    axs[2].set_title('Після згладжування МНК')


    # Додаткові налаштування
    for ax in axs:
        ax.legend()
        ax.set_xlabel('Похибка')
        ax.set_ylabel('Частота')


    plt.suptitle('Гістограми похибок')
    plt.show()


    # Виведення таблички у консоль
    print("\nСтатистичні характеристики:")
    print(statistics_table)




# Закон зміни похибки – експонентційний;
def exp_d(size=100000, lambda_val=0.1):
    """ Exponential Distribution array generator """
    return np.random.exponential(scale=1.0 / lambda_val, size=size)




# Генерація експериментальної вибірки
np.random.seed(42)

n_measurements = 1000
percentage_anomalies = 0.1

ideal_values = np.linspace(0, 10, n_measurements) ** 2
#noise = np.random.normal(0, 1, n_measurements)
noise = exp_d(n_measurements, 4)

anomalies = np.random.choice(n_measurements, int(n_measurements * percentage_anomalies), replace=False)
measurements = ideal_values + noise + 3 * np.std(noise) * (np.random.rand(n_measurements) - 0.5)
measurements[anomalies] += 10  # Додавання аномальних вимірів

# Алгоритм виявлення аномальних вимірів: Метод найменших квадратів
anomalies_detected_var1 = detect_anomalies(measurements)
# Метод усунення впливу аномальних вимірів: Відновлення вимірів
restored_measurements_var1 = restore_measurements(measurements, anomalies_detected_var1)
# Здійснення згладжування методом найменших квадратів
smoothed_data_var1 = smooth_data(measurements, anomalies_detected_var1)
# Монте-Карло аналіз
monte_carlo_results_var1 = monte_carlo_analysis(measurements, anomalies_detected_var1)