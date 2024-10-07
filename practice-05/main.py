import numpy as np
from scipy.stats import norm
from scipy.stats import median_abs_deviation as mad


# Функция для бутстрепа
def bootstrap_mad(data, n_iterations=100, sample_size=10000):
    bootstrapped_mads = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=sample_size, replace=True)
        bootstrapped_mads.append(mad(sample))
    return np.std(bootstrapped_mads)


# Функция для вычисления асимптотического доверительного интервала
def asymptotic_confidence_interval(mad_star, s_mad_star, confidence=0.95):
    z = norm.ppf((1 + confidence) / 2)
    return mad_star - s_mad_star * z, mad_star + s_mad_star * z


# Функция для вычисления приближенного доверительного интервала (большие n)
def large_n_approximation(data, confidence=0.95):
    n = len(data)
    z = norm.ppf((1 + confidence) / 2)
    k1 = int(0.5 * n - 0.5 * np.sqrt(n) * z)
    k2 = int(0.5 * n + 0.5 * np.sqrt(n) * z) + 1
    sorted_data = np.sort(abs(data - np.median(data)))
    return sorted_data[k1], sorted_data[k2]


def small_n_approximation(data, confidence=0.95):
    n = len(data)
    z = norm.ppf((1 + confidence) / 2)
    # Индексы k1 и k2 для доверительных интервалов
    k1 = int(0.5 * n - 0.5 * np.sqrt(n) * z)
    k2 = int(0.5 * n + 0.5 * np.sqrt(n) * z) + 1
    # Сортируем данные
    sorted_data = np.sort(data)
    # Граничные значения x_k1 и x_k2
    x_k1, x_k2 = sorted_data[k1], sorted_data[k2]
    # Рассчитываем минимальные и максимальные расстояния
    y_1 = np.minimum(np.abs(data - x_k1), np.abs(data - x_k2))
    y_2 = np.maximum(np.abs(data - x_k1), np.abs(data - x_k2))
    # Сортируем минимальные и максимальные расстояния
    sorted_y_1 = sorted(y_1)
    sorted_y_2 = sorted(y_2)
    # Возвращаем отсортированные минимальные и максимальные границы
    return sorted_y_1[k1], sorted_y_2[k2]


# Генерация распределений и вычисление точного значения MAD для большого размера выборки
large_size = 10 ** 7
uniform_dist = np.random.uniform(-1, 1, large_size)
normal_dist = np.random.normal(0, 1, large_size)
sum_uniform_dist = np.random.uniform(-1, 1, large_size) + np.random.uniform(-1, 1, large_size)

# Точные значения MAD
mad_uniform = mad(uniform_dist)
mad_normal = mad(normal_dist)
mad_sum_uniform = mad(sum_uniform_dist)

# Меньшие размеры выборки
sample_sizes = [10, 100, 500, 1000, 10000]

# Confidence level
confidence_level = 0.95

# Для каждого распределения и размера выборки
results = {}
distributions = {
    'Uniform(-1, 1)': uniform_dist,
    'Normal(0, 1)': normal_dist,
    'Sum of two Uniform(-1, 1)': sum_uniform_dist
}

for dist_name, dist_data in distributions.items():
    results[dist_name] = {}
    mad_star = mad(dist_data)
    s_mad_star = bootstrap_mad(dist_data)  # Оценка стандартного отклонения с бутстрепом

    for n in sample_sizes:
        sample = np.random.choice(dist_data, n, replace=False)

        # Асимптотический доверительный интервал
        asymptotic_interval = asymptotic_confidence_interval(mad_star, s_mad_star, confidence=confidence_level)

        # Приближенные доверительные интервалы
        large_n_interval = large_n_approximation(sample, confidence=confidence_level)
        small_n_interval = small_n_approximation(sample, confidence=confidence_level)

        results[dist_name][f' Sample size {n}'] = {
            'Asymptotic CI': asymptotic_interval,
            'Large n Approximation': large_n_interval,
            'Small n Approximation ': small_n_interval,
        }
# Вывод результатов
for dist_name, sample_sizes in results.items():
    print(f"Distribution: {dist_name}")
    for sample_size, intervals in sample_sizes.items():
        print(f"  {sample_size}:")
        for interval_type, interval in intervals.items():
            print(f"    {interval_type}: {interval}")
    print()
