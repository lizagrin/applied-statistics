import numpy as np
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt
import time


def generate_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size)


def generate_normal(size):
    return np.random.normal(0, 1, size)


def generate_sum_2_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size) + np.random.uniform(-1, 1, size)


def precompute_binomial_coefficients(n, p=0.5):
    # Предварительное вычисление биномиальных коэффициентов
    coeffs = np.array([comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(n + 1)])
    return np.cumsum(coeffs)


def exact_confidence_interval(n, p=0.5, Q=0.95):
    # Предварительное вычисление биномиальных коэффициентов
    cumulative_probs = precompute_binomial_coefficients(n, p)

    # Определение границ доверительного интервала
    m1 = np.searchsorted(cumulative_probs, (1 - Q) / 2)
    m2 = np.searchsorted(cumulative_probs, 1 - (1 - Q) / 2)

    return m1, m2


def approximate_confidence_interval(n, p=0.5, Q=0.95):
    # Функция для вычисления приближенного доверительного интервала с использованием нормального распределения
    z = norm.ppf((1 + Q) / 2)  # Квантиль стандартного нормального распределения
    # Вычисление нижней и верхней границ интервала
    m1 = int(np.floor(n * p - np.sqrt(n * p * (1 - p)) * z))
    m2 = int(np.ceil(n * p + np.sqrt(n * p * (1 - p)) * z))
    return m1, m2


def analyze_distribution(generate_data_func, sample_sizes, distribution_name, num_simulations=10000):
    true_median = 0

    # Списки для хранения вероятностей покрытия для точного и приближенного интервалов
    exact_coverages = []
    approx_coverages = []

    # Перебор различных размеров выборки
    for N in sample_sizes:
        exact_coverage_count = 0  # Счетчик для точного интервала
        approx_coverage_count = 0  # Счетчик для приближенного интервала
        exact_time_total = 0
        approx_time_total = 0

        for _ in range(num_simulations):
            # Генерация данных с использованием переданной функции
            data = generate_data_func(N)
            sorted_data = np.sort(data)  # Сортировка данных

            start_time = time.time()
            # Вычисление индексов для точного доверительного интервала
            m1_exact, m2_exact = exact_confidence_interval(N)
            exact_time_total += time.time() - start_time
            ci_exact = (sorted_data[m1_exact], sorted_data[m2_exact])

            start_time = time.time()
            # Вычисление индексов для приближенного доверительного интервала
            m1_approx, m2_approx = approximate_confidence_interval(N)
            approx_time_total += time.time() - start_time
            ci_approx = (sorted_data[m1_approx], sorted_data[m2_approx])

            # Проверка, покрывает ли доверительный интервал истинную медиану
            if ci_exact[0] <= true_median <= ci_exact[1]:
                exact_coverage_count += 1
            if ci_approx[0] <= true_median <= ci_approx[1]:
                approx_coverage_count += 1

        # Вычисление вероятностей покрытия
        exact_coverage_probability = exact_coverage_count / num_simulations
        approx_coverage_probability = approx_coverage_count / num_simulations

        # Добавление вероятностей покрытия в списки
        exact_coverages.append(exact_coverage_probability)
        approx_coverages.append(approx_coverage_probability)

        # Печать результатов для текущего размера выборки
        print(
            f"{distribution_name}: Size: {N}, Median: {true_median}, Exact CI: {ci_exact}, Approx CI: {ci_approx}")
        print(f"Exact method time: {exact_time_total:.4f} seconds")
        print(f"Approximate method time: {approx_time_total:.4f} seconds")
    # Определение доверительных интервалов для вероятностей покрытия
    q = 0.95
    z_score = norm.ppf((1 + q) / 2)  # z-оценка для нормального распределения

    # Вычисление доверительных интервалов для точных и приближенных вероятностей покрытия
    exact_intervals = [(p - z_score * np.sqrt(p * (1 - p) / num_simulations),
                        p + z_score * np.sqrt(p * (1 - p) / num_simulations)) for p in exact_coverages]
    approx_intervals = [(p - z_score * np.sqrt(p * (1 - p) / num_simulations),
                         p + z_score * np.sqrt(p * (1 - p) / num_simulations)) for p in approx_coverages]

    # Печать доверительных интервалов
    print(
        f"\n {distribution_name}: Exact intervals: {exact_intervals}, Approx intervals: {approx_intervals}")

    # Построение графика вероятностей покрытия
    plt.figure(figsize=(10, 6))
    offset = 0.7  # Смещение для разделения линий на графике

    for i, N in enumerate(sample_sizes):
        # Линии для точных доверительных интервалов
        plt.plot([N - offset, N - offset], [exact_intervals[i][0], exact_intervals[i][1]], color='blue', linestyle='-')
        plt.plot(N, exact_coverages[i], 'o', color='blue', label='Exact CI Coverage' if i == 0 else "")

        # Линии для приближенных доверительных интервалов
        plt.plot([N + offset, N + offset], [approx_intervals[i][0], approx_intervals[i][1]], color='red', linestyle='-')
        plt.plot(N, approx_coverages[i], 'o', color='red', label='Approx CI Coverage' if i == 0 else "")

    # Линия желаемого уровня покрытия (0.95)
    plt.axhline(y=0.95, color='green', linestyle='--', label='Desired Coverage (0.95)')

    # Настройка заголовка и меток осей
    plt.title(f'Coverage Probability for {distribution_name}')
    plt.xlabel('Sample Size')
    plt.ylabel('Coverage Probability')
    plt.ylim(0.9, 1.0)
    plt.legend()
    plt.grid(True)
    plt.show()


# Заданные размеры выборок
sample_sizes = [10, 100, 300, 500, 700, 1000]

for distribution_func, name in [
    (generate_uniform_minus1_1, "Uniform [-1, 1]"),
    (generate_normal, "Normal (0, 1)"),
    (generate_sum_2_uniform_minus1_1, "Sum of Uniforms")
]:
    analyze_distribution(distribution_func, sample_sizes, name)
