import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Параметры моделирования
M = 1000  # Количество выборок
sample_sizes = [10, 20, 50, 100, 200]  # Размеры выборок
alpha = 0.05  # Порог FRR


# Генерация выборок
def generate_samples(distribution, n, M):
    if distribution == "normal":
        return np.random.normal(0, 1, (M, n))
    elif distribution == "uniform":
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), (M, n))  # Генерация на [-sqrt(3), sqrt(3)]
    elif distribution == "exponential":
        return np.random.exponential(1, (M, n)) - 1


# Z-score
def z_score_method(samples):
    means = np.mean(samples, axis=1, keepdims=True)
    stds = np.std(samples, axis=1, keepdims=True)
    z_scores = (samples - means) / stds
    return np.abs(z_scores)


# IQR и boxplot
def iqr_method(samples, k):
    q1 = np.percentile(samples, 25, axis=1)
    q3 = np.percentile(samples, 75, axis=1)
    iqr = q3 - q1
    lower_bound = q1[:, np.newaxis] - k * iqr[:, np.newaxis]
    upper_bound = q3[:, np.newaxis] + k * iqr[:, np.newaxis]
    return (samples < lower_bound) | (samples > upper_bound)


# DBSCAN
def dbscan_method(samples, eps, min_samples=5):
    outliers = np.zeros(samples.shape, dtype=bool)
    for i in range(samples.shape[0]):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(samples[i].reshape(-1, 1))
        outliers[i] = clustering.labels_ == -1
    return outliers


# Функция для адаптивного подбора k
def find_adaptive_k(samples, target_alpha, initial_left=0, initial_right=3.0, tolerance=0.00001):
    left, right = initial_left, initial_right
    while right - left > tolerance:
        mid = (left + right) / 2
        iqr_outliers = iqr_method(samples, mid)
        outlier_fraction = np.mean(iqr_outliers)
        if outlier_fraction < target_alpha:
            right = mid
        else:
            left = mid
    return (left + right) / 2


# Адаптивный подбор eps для DBSCAN
def find_adaptive_eps(samples, target_alpha, min_samples=5, initial_left=0.01, initial_right=1.0, tolerance=0.00001):
    left, right = initial_left, initial_right
    while right - left > tolerance:
        mid = (left + right) / 2
        dbscan_outliers = dbscan_method(samples, eps=mid, min_samples=min_samples)
        outlier_fraction = np.mean(dbscan_outliers)
        if outlier_fraction < target_alpha:
            right = mid
        else:
            left = mid
    return (left + right) / 2


# Численное моделирование для определения констант
def calculate_constants(distribution, alpha, n, M):
    samples = generate_samples(distribution, n, M)

    # Z-score threshold
    z_scores = z_score_method(samples)
    z_threshold = np.percentile(z_scores.flatten(), 100 * (1 - alpha))

    # Адаптивный подбор коэффициента k для IQR
    iqr_k = find_adaptive_k(samples, alpha)

    # Адаптивный подбор eps для DBSCAN
    eps = find_adaptive_eps(samples, alpha, min_samples=5)

    return z_threshold, iqr_k, eps


# Для хранения результатов
results = {distribution: {'z': [], 'iqr': [], 'dbscan': [], 'at_least_two': []} for distribution in
           ["normal", "uniform", "exponential"]}

# Основной цикл
for n in sample_sizes:
    for distribution in ["normal", "uniform", "exponential"]:
        z_threshold, iqr_k, eps = calculate_constants(distribution, alpha, n, M)
        samples = generate_samples(distribution, n, M)

        # Вычисляем выбросы по каждому методу
        z_outliers = z_score_method(samples) > z_threshold
        iqr_outliers = iqr_method(samples, iqr_k)
        dbscan_outliers = dbscan_method(samples, eps=eps, min_samples=5)

        # Считаем количество выбросов для каждого метода
        z_outliers_count = np.mean(np.sum(z_outliers, axis=1))
        iqr_outliers_count = np.mean(np.sum(iqr_outliers, axis=1))
        dbscan_outliers_count = np.mean(np.sum(dbscan_outliers, axis=1))

        # Подсчет выбросов, которые определены хотя бы двумя методами
        at_least_two = np.mean(np.sum((z_outliers & iqr_outliers) |
                                      (z_outliers & dbscan_outliers) |
                                      (iqr_outliers & dbscan_outliers), axis=1))

        # Сохраняем результаты
        results[distribution]['z'].append(z_outliers_count / n * 100)
        results[distribution]['iqr'].append(iqr_outliers_count / n * 100)
        results[distribution]['dbscan'].append(dbscan_outliers_count / n * 100)
        results[distribution]['at_least_two'].append(at_least_two / n * 100)

# Построение графиков
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
distributions = ["normal", "uniform", "exponential"]

for i, distribution in enumerate(distributions):
    axes[i].plot(sample_sizes, results[distribution]['z'], label='Z-score')
    axes[i].plot(sample_sizes, results[distribution]['iqr'], label='IQR')
    axes[i].plot(sample_sizes, results[distribution]['dbscan'], label='DBSCAN')
    axes[i].plot(sample_sizes, results[distribution]['at_least_two'], label='At least two methods', linestyle='--')
    axes[i].set_title(f'{distribution.capitalize()} Distribution')
    axes[i].set_xlabel('Sample Size (n)')
    axes[i].set_ylabel('Outliers Percentage (%)')
    axes[i].legend()

plt.suptitle('Outliers Percentage by Method and Sample Size')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
