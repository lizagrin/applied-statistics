import numpy as np
from scipy.stats import norm
from scipy.special import comb


def generate_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size)


def generate_normal(size):
    return np.random.normal(0, 1, size)


def generate_sum_2_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size) + np.random.uniform(-1, 1, size)


def exact_confidence_interval(n, p=0.5, Q=0.95):
    # Функция для вычисления точного доверительного интервала для биномиального распределения
    m1 = m2 = None  # Инициализация границ интервала
    for m in range(n + 1):
        # Вычисление вероятности, что случайная величина <= m
        prob = sum(comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(m + 1))
        # Установка нижней границы интервала
        if prob >= (1 - Q) / 2 and m1 is None:
            m1 = m
        # Установка верхней границы интервала и выход из цикла
        if prob >= 1 - (1 - Q) / 2:
            m2 = m
            break
    return m1, m2


def approximate_confidence_interval(n, p=0.5, Q=0.95):
    # Функция для вычисления приближенного доверительного интервала с использованием нормального распределения
    z = norm.ppf((1 + Q) / 2)  # Квантиль стандартного нормального распределения
    # Вычисление нижней и верхней границ интервала
    m1 = int(np.floor(n * p - np.sqrt(n * p * (1 - p)) * z))
    m2 = int(np.ceil(n * p + np.sqrt(n * p * (1 - p)) * z))
    return m1, m2


def analyze_distribution(generate_data_func, N, distribution_name):
    # Функция для анализа распределения и вычисления медианы и доверительных интервалов
    data = generate_data_func(N)  # Генерация данных с помощью переданной функции
    median = 0  # Инициализация медианы

    # Вычисление точного доверительного интервала
    m1_exact, m2_exact = exact_confidence_interval(N)
    ci_exact = (np.sort(data)[m1_exact], np.sort(data)[m2_exact])

    # Вычисление приближенного доверительного интервала
    m1_approx, m2_approx = approximate_confidence_interval(N)
    ci_approx = (np.sort(data)[m1_approx], np.sort(data)[m2_approx])

    print(f"{distribution_name}: Median: {median}, Exact CI: {ci_exact}, Approx CI: {ci_approx}")


sample_sizes = [10, 100, 500, 1000]

for N in sample_sizes:
    print(f"\n Sample size: {N}")

    analyze_distribution(generate_uniform_minus1_1, N, "Uniform [-1, 1]")
    analyze_distribution(generate_normal, N, "Normal (0, 1)")
    analyze_distribution(generate_sum_2_uniform_minus1_1, N, "Sum of Uniforms")
