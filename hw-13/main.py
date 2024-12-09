import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Параметры задачи
N = 100  # Число сигналов
n = 100  # Число точек в каждом сигнале
t = np.arange(n)  # Моменты времени
base_signal = 5 * np.exp(-2 * t / n)  # Сигнал x(t) = 5 * exp(-2 * t / n)


# Генерация сигналов для каждого распределения
def generate_signals(distribution, N, base_signal):
    if distribution == "normal":
        signals = np.random.normal(size=(N, len(base_signal)))  # Нормальное распределение
    elif distribution == "uniform":
        signals = np.random.uniform(-1, 1, size=(N, len(base_signal)))  # Равномерное распределение
    elif distribution == "exponential":
        signals = np.random.exponential(scale=1, size=(N, len(base_signal)))  # Экспоненциальное распределение
    return signals * base_signal  # Наложение распределения на базовый сигнал


# Вычисление автокорреляционной функции
def autocorrelation(signal):
    n = len(signal)
    result = np.correlate(signal, signal, mode='full') / n
    return result[n - 1:] / result[n - 1]


# Функция для нахождения tau
def find_tau(autocorr, stderr, z):
    for tau in range(1, len(autocorr)):  # Tau > 0
        if abs(autocorr[tau]) <= z * stderr:  # Проверяем границы доверительного интервала
            return tau
    return len(autocorr)  # Если не найдено, возвращаем максимальное значение tau


# Генерация сигналов для каждого распределения
distributions = ["normal", "uniform", "exponential"]
signals_data = {dist: generate_signals(dist, N, base_signal) for dist in distributions}

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
        find_tau(autocorr, stderr=np.std(autocorr), z=z)
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
