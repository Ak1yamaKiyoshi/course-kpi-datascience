import numpy as np
import matplotlib.pyplot as plt

# Define parameters
amplitude = 1
frequency = 1
phase = 0
mean = 0
std_dev = 0.1

# Generate time points
time = np.linspace(0, 2*np.pi, 1000)

# Calculate periodic function values
values = amplitude * np.sin(frequency * time + phase)

# Generate normally distributed noise
noise = np.random.normal(mean, std_dev, time.shape)

# Add noise to values
noisy_values = values + noise

# Plot original and noisy values
plt.figure(figsize=(10, 6))
plt.plot(time, values, label='Original')
plt.plot(time, noisy_values, label='Noisy', linestyle='dashed')
plt.legend()
plt.show()