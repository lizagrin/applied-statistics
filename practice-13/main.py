import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Параметры задачи
N = 100  # Число сигналов
n = 100  # Число точек в каждом сигнале
noise_range = 0.2  # Максимальная случайная погрешность

# Генерация сигналов
def generate_signals(distribution, N, n):
    if distribution == "normal":
        signals = np.random.randn(N, n)  # Нормальное распределение
    elif distribution == "uniform":
        signals = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(N, n))  # Равномерное
    elif distribution == "exponential":
        signals = np.random.exponential(scale=1, size=(N, n))  # Экспоненциальное
    # Нормировка до нулевого МО и единичной дисперсии
    signals = (signals - np.mean(signals, axis=1, keepdims=True)) / np.std(signals, axis=1, keepdims=True)
    return signals

# Добавление погрешности
def add_noise(signals, noise_range):
    noise = np.random.uniform(-noise_range, noise_range, size=signals.shape)
    return signals + noise

# Вычисление автокорреляционной функции
def autocorrelation(signal):
    n = len(signal)
    result = np.correlate(signal, signal, mode='full') / n
    return result[n - 1:] / result[n - 1]

# Функция для нахождения tau
def find_tau(autocorr, stderr, z):
    """
    Находит минимальное значение tau > 0, где автокорреляция входит в доверительный интервал [-z*stderr, z*stderr].
    """
    for tau in range(1, len(autocorr)):  # Tau > 0
        if abs(autocorr[tau]) <= z * stderr[tau]:  # stderr зависит от tau
            return tau
    return len(autocorr)  # Если не найдено, возвращаем максимальное значение tau

# Генерация сигналов для каждого распределения
distributions = ["normal", "uniform", "exponential"]
signals_data = {dist: add_noise(generate_signals(dist, N, n), noise_range) for dist in distributions}

# Вычисление автокорреляции
autocorrelations = {dist: np.array([autocorrelation(sig) for sig in signals]) for dist, signals in signals_data.items()}

# Построение доверительных интервалов
confidence_level = 0.95
z = norm.ppf((1 + confidence_level) / 2)

plt.figure(figsize=(15, 10))
for i, (dist, autocorr) in enumerate(autocorrelations.items(), 1):
    mean_autocorr = np.mean(autocorr, axis=0)
    std_autocorr = np.std(autocorr, axis=0)

    # Доверительные интервалы для каждой точки
    lower_bound = mean_autocorr - z * std_autocorr
    upper_bound = mean_autocorr + z * std_autocorr
    lags = np.arange(len(mean_autocorr))

    # Построение графиков
    plt.subplot(3, 1, i)
    plt.plot(lags, mean_autocorr, label="Mean autocorrelation")
    plt.fill_between(lags, lower_bound, upper_bound, color='gray', alpha=0.3, label="95% Confidence Interval")
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label="y = 0")
    plt.title(f"Autocorrelation for {dist.capitalize()} distribution")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()

plt.tight_layout()
plt.show()

# Поиск значений tau для каждого сигнала
taus = {
    dist: [
        find_tau(autocorr, stderr=np.std(autocorr, axis=0), z=z)
        for autocorr in autocorrelations[dist]
    ]
    for dist in distributions
}

# Построение эмпирической функции распределения для каждого распределения
plt.figure(figsize=(15, 10))
for i, (dist, tau_values) in enumerate(taus.items(), 1):
    sorted_taus = np.sort(tau_values)
    cdf = np.arange(1, len(sorted_taus) + 1) / len(sorted_taus)  # Эмпирическая функция распределения
    plt.subplot(3, 1, i)
    plt.step(sorted_taus, cdf, where='post', label=f"CDF for {dist.capitalize()} distribution")
    plt.xlabel("Tau (lag)")
    plt.ylabel("Empirical CDF")
    plt.title(f"Empirical distribution of Tau for {dist.capitalize()} distribution")
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()
