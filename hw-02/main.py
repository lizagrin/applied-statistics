import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Функция для генерации выборок из заданного распределения
def generate_samples(distribution, sample_size, num_samples=10000):
    if distribution == 'uniform':
        return np.random.uniform(-1, 1, (num_samples, sample_size))
    elif distribution == 'normal':
        return np.random.normal(0, 1, (num_samples, sample_size))
    elif distribution == 'sum_uniform':
        return np.random.uniform(-1, 1, (num_samples, sample_size)) + np.random.uniform(-1, 1,
                                                                                        (num_samples, sample_size))
    else:
        raise ValueError("Неизвестное распределение")


# Функция для вычисления доверительного интервала методом бутстрапа
def bootstrap_confidence_interval(data, num_bootstrap=1000, alpha=0.05):
    means = []  # Список для хранения средних значений бутстрапированных выборок
    n = len(data)
    for _ in range(num_bootstrap):
        # Создание бутстрапированной выборки с возвращением
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))  # Вычисление среднего значения бутстрапированной выборки
    lower = np.percentile(means, 100 * alpha / 2)  # Нижняя граница доверительного интервала
    upper = np.percentile(means, 100 * (1 - alpha / 2))  # Верхняя граница доверительного интервала
    return lower, upper, means


# Функция для вычисления доверительного интервала методом джекнайфа
def jackknife_confidence_interval(data, alpha=0.05):
    n = len(data)
    # Вычисление средних значений выборок
    jackknife_means = np.array([np.mean(np.delete(data, i)) for i in range(n)])
    jackknife_mean = np.mean(jackknife_means)  # Среднее значение
    jackknife_var = (n - 1) * np.var(jackknife_means)  # Оценка дисперсии
    se = np.sqrt(jackknife_var)  # Стандартная ошибка
    lower = jackknife_mean - norm.ppf(1 - alpha / 2) * se  # Нижняя граница доверительного интервала
    upper = jackknife_mean + norm.ppf(1 - alpha / 2) * se  # Верхняя граница доверительного интервала
    return lower, upper, jackknife_means


# Функция для построения графиков доверительных интервалов
def plot_confidence_intervals(sample, distribution_name, size):
    true_mean = 0  # Истинное среднее значение

    # Вычисление доверительных интервалов методом бутстрапа
    lower_b, upper_b, bootstrap_means = bootstrap_confidence_interval(sample)

    # Вычисление доверительных интервалов методом джекнайфа
    lower_j, upper_j, jackknife_means = jackknife_confidence_interval(sample)

    plt.figure(figsize=(14, 6))

    # График для бутстрапа
    plt.subplot(1, 2, 1)
    plt.hist(bootstrap_means, bins=30, alpha=0.7, color='skyblue', label='Bootstrap Means')
    plt.axvline(lower_b, color='red', linestyle='--', label='Bootstrap CI Lower')
    plt.axvline(upper_b, color='red', linestyle='--', label='Bootstrap CI Upper')
    plt.axvline(true_mean, color='green', linestyle='-', label='True Mean')
    plt.title(f'Bootstrap CI for {distribution_name} with n={size}')
    plt.xlabel('Mean')
    plt.ylabel('Frequency')
    plt.legend()

    # График для джекнайфа
    plt.subplot(1, 2, 2)
    plt.hist(jackknife_means, bins=30, alpha=0.7, color='lightcoral', label='Jackknife Means')
    plt.axvline(lower_j, color='red', linestyle='--', label='Jackknife CI Lower')
    plt.axvline(upper_j, color='red', linestyle='--', label='Jackknife CI Upper')
    plt.axvline(true_mean, color='green', linestyle='-', label='True Mean')
    plt.title(f'Jackknife CI for {distribution_name} with n={size}')
    plt.xlabel('Mean')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Определение распределений и размеров выборок
distributions = ['uniform', 'normal', 'sum_uniform']
sample_sizes = [1000, 5000]

# Генерация и построение графиков для каждого распределения и размера выборки
for distribution in distributions:
    for size in sample_sizes:
        samples = generate_samples(distribution, size)
        sample = samples[0]  # Берем первую выборку для визуализации
        plot_confidence_intervals(sample, distribution, size)
