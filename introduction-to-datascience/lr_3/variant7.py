import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def alpha_beta_filter(data, alpha, beta):
    estimation, velocity = data[0], 0
    estimations = []

    for measurement in data:
        estimation = estimation + velocity
        residual = measurement - estimation
        estimation = estimation + alpha * residual
        velocity = velocity + beta * residual / 1
        estimations.append(estimation)

    return estimations

def run(measurements):
    alpha, beta = 0.2, 0.1
    filtered_data = alpha_beta_filter(measurements, alpha, beta)
    return np.array(filtered_data)


def detect_anomalies_least_squares(data, threshold_factor=2.0):
    """
    Виявлення аномалій методом найменших квадратів.

    Params:
    - data: numpy array, часовий ряд вимірювань
    - threshold_factor: множник порогового значення

    Returns:
    - anomalies: список індексів аномальних вимірювань
    """
    n = len(data)
    X = np.column_stack((np.arange(1, n + 1), np.ones(n)))
    y = data.reshape(-1, 1)
    beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - np.dot(X, beta)
    std_residuals = np.std(residuals)
    anomalies = np.where(np.abs(residuals) > threshold_factor * std_residuals)[0]
    return anomalies

def reject_anomalies(data, anomalies):
    """
    Метод усунення впливу аномальних вимірів (відкидання).

    Parameters:
    - data: numpy array, вхідні виміри
    - anomalies: numpy array, індекси аномальних вимірів

    Returns:
    - cleaned_data: numpy array, виміри після усунення аномалій
    """
    cleaned_data = np.delete(data, anomalies)
    return cleaned_data

def monte_carlo_analysis(noisy_data, anomalies, iterations=1000):
    """ anomalies - indicies of anomalies """
    errors_noisy = []
    errors_anomalous = []
    for _ in range(iterations):
        noisy_data_iteration = noisy_data + np.random.normal(0, 1, len(noisy_data))
        errors_noisy.append(np.mean((noisy_data_iteration - noisy_data) ** 2))
        anomalous_data = np.copy(noisy_data_iteration)
        anomalous_data[anomalies] += 10
        errors_anomalous.append(np.mean((anomalous_data - noisy_data) ** 2))

    return {
        'mean_noisy': np.mean(errors_noisy),
        'std_dev_noisy': np.std(errors_noisy),
        'mean_anomalous': np.mean(errors_anomalous),
        'std_dev_anomalous': np.std(errors_anomalous),
    }

np.random.seed(42)
# CONSTANTS
MESAURMENTS = 100
PERCENTAGE_ANOMALIES = 0.1

# MODELS
""" Квадратичний тренд """
model_ideal = np.linspace(0, 10, MESAURMENTS) ** 2

""" Шум нормального розпоілу """
noise = np.random.normal(loc=0.0, scale=1.0, size=MESAURMENTS)

""" Зашумлені виміри без аномалій """
measurements = model_ideal + noise + 3 * np.std(noise) * (np.random.rand(MESAURMENTS) - 0.5)

""" Аномалії """
anomalies = np.random.choice(MESAURMENTS, int(MESAURMENTS * PERCENTAGE_ANOMALIES), replace=False)

""" Зашумлені виміри з аномаліями """
measurments_with_anomalies = measurements.copy()
measurments_with_anomalies[anomalies] += 10

""" Відновлені від аномалій виміри шляхом відкидання """
restored_measurments_rejected = reject_anomalies(measurements, anomalies)

""" Згладжені виміри відновлених вимірів """
smoothed_restored_reject_measurments = run(restored_measurments_rejected)

""" Злагоджені виміри вимірів без аномалій """
smoothed_without_anomalies = run(measurements)

statistics_table = pd.DataFrame({
    'Параметр': ['Математичне сподівання', 'Середньоквадратичне відхилення'],
    'Шум': [np.mean(noise), np.std(noise)],
    'З аномаліями': [np.mean(measurments_with_anomalies), np.std(measurments_with_anomalies)],
    'Після згладжування Фільтром alpha-beta без аномалій': [np.mean(smoothed_without_anomalies), np.std(smoothed_without_anomalies)],
    'Після згладжування Фільтром alpha-beta з відновленням (reject)': [np.mean(smoothed_restored_reject_measurments), np.std(smoothed_restored_reject_measurments)],
})

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(statistics_table)
    
import pprint
pprint.pprint(
monte_carlo_analysis(measurements, anomalies)
)

""" Графіки даних """
""" Ідеальна та зашумлені моделі """
plt.figure(figsize=(8, 6))

plt.subplot(3, 1, 1)
plt.plot(model_ideal, label='Квадратичний тренд', color='blue')
plt.legend()
plt.title('Квадратичний тренд')

plt.subplot(3, 1, 2)
plt.plot(measurements, label='Зашумлена без аномалій', color='orange')
plt.legend()
plt.title('Зашумлена без аномалій')

plt.subplot(3, 1, 3)
plt.legend()
plt.title('Зашумлена з аномаліями')
plt.plot(measurments_with_anomalies, label='Зашумлена з аномаліями', color='green')

plt.tight_layout()
plt.show()


""" Злагоджені моделі """
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(smoothed_restored_reject_measurments, label='Результат згладжування Фільтром alpha-beta відновлених методом відкидання', linestyle='--', color='purple')
plt.legend()
plt.title('Результат згладжування Фільтром alpha-beta відновлених методом відкидання')

valid_anomalies = anomalies[anomalies < len(measurments_with_anomalies)]
plt.scatter(valid_anomalies, measurments_with_anomalies[valid_anomalies], color='red', label='Аномалії')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(smoothed_without_anomalies, label='Результат згладжування Фільтром alpha-beta зашумлених даних без аномалій', linestyle='--', color='purple')
plt.legend()
plt.title('Результат згладжування Фільтром alpha-beta зашумлених даних без аномалій')

valid_anomalies = anomalies[anomalies < len(measurments_with_anomalies)]
plt.scatter(valid_anomalies, measurments_with_anomalies[valid_anomalies], color='red', label='Аномалії')
plt.legend()

# Додаткові налаштування
plt.tight_layout()
plt.show()



""" Гістограми розподілу """
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(measurements, alpha=0.5, color='blue', label='Без аномалій')
axs[0].set_title('Без аномалій')

axs[1].hist(measurments_with_anomalies, alpha=0.5, color='red', label='З аномаліями')
axs[1].set_title('З аномаліями')

axs[2].hist(model_ideal, alpha=0.5, color='green', label='Another Data')
axs[2].set_title('Ідеальна модель')

plt.tight_layout()
plt.show()


""" Гістограми розподілу злагодження Фільтром alpha-beta """
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].hist(smoothed_without_anomalies, alpha=0.5, color='blue', label='Злагоджене без аномалій')
axs[0].set_title('Злагоджене без аномалій')

axs[1].hist(smoothed_restored_reject_measurments, alpha=0.5, color='green', label='Злагоджене з аномаліями ( метод відкидання )')
axs[1].set_title('Злагоджене з аномаліями ( метод відкидання ) ')

plt.tight_layout()
plt.show()