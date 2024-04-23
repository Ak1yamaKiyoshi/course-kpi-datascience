import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


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
    restored_data[anomalies] = np.nan
    valid_indices = np.where(~np.isnan(restored_data))[0]
    restored_data[anomalies] = np.polyval(np.polyfit(valid_indices, restored_data[valid_indices], 2), anomalies)
    return restored_data

def run(measurements):
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []

    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    return np.array(predictions)


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

""" Відновлені від аномалій вміри шляхом відновлення """
restored_measurments_restored = restore_measurements(measurements, anomalies)

""" Згладжені виміри відновлених вимірів """
smoothed_restored_restore_measurments= run(restored_measurments_restored)

""" Злагоджені виміри вимірів без аномалій """
smoothed_without_anomalies = run(measurements)



statistics_table = pd.DataFrame({
    'Параметр': ['Математичне сподівання', 'Середньоквадратичне відхилення'],
    'Шум': [np.mean(noise), np.std(noise)],
    'З аномаліями': [np.mean(measurments_with_anomalies), np.std(measurments_with_anomalies)],
    'Після згладжування Калманом без аномалій': [np.mean(smoothed_without_anomalies), np.std(smoothed_without_anomalies)],
    'Після згладжування Калманом з відновленням (restore)': [np.mean(smoothed_restored_restore_measurments), np.std(smoothed_restored_restore_measurments)],
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
plt.plot(smoothed_restored_restore_measurments, label='Результат згладжування Калмана відновлених методом відновлення', linestyle='--', color='purple')
plt.legend()
plt.title('Результат згладжування Калмана відновлених методом відновлення')

valid_anomalies = anomalies[anomalies < len(measurments_with_anomalies)]
plt.scatter(valid_anomalies, measurments_with_anomalies[valid_anomalies], color='red', label='Аномалії')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(smoothed_without_anomalies, label='Результат згладжування Калмана зашумлених даних без аномалій', linestyle='--', color='purple')
plt.legend()
plt.title('Результат згладжування Калмана зашумлених даних без аномалій')

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



""" Гістограми розподілу злагодження Калмана """
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(smoothed_without_anomalies, alpha=0.5, color='blue', label='Злагоджене без аномалій')
axs[0].set_title('Злагоджене без аномалій')

axs[1].hist(smoothed_restored_restore_measurments, alpha=0.5, color='red', label='Злагоджене з аномаліями ( метод відновлення )')
axs[1].set_title('Злагоджене з аномаліями ( метод відновлення ) ')

plt.tight_layout()
plt.show()