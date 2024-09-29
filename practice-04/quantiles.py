import numpy as np
from scipy.stats import norm
from scipy.special import comb


def generate_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size)


def generate_normal(size):
    return np.random.normal(0, 1, size)


def generate_sum_2_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size) + np.random.uniform(-1, 1, size)


# Предварительное вычисление биномиальных коэффициентов для заданного n и p
def precompute_binomial_coefficients(n, p=0.5):
    coeffs = np.array([comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(n + 1)])
    return np.cumsum(coeffs)


# Вычисление точного доверительного интервала для биномиального распределения
def exact_confidence_interval(n, p=0.5, Q=0.95):
    cumulative_probs = precompute_binomial_coefficients(n, p)
    m1 = np.searchsorted(cumulative_probs, (1 - Q) / 2)
    m2 = np.searchsorted(cumulative_probs, 1 - (1 - Q) / 2)

    # Ограничение индексов в пределах размера выборки
    m1 = max(0, min(m1, n - 1))
    m2 = max(0, min(m2, n - 1))

    return m1, m2


# Вычисление аппроксимированного доверительного интервала используя нормальное распределение
def approximate_confidence_interval(n, p=0.5, Q=0.95):
    z = norm.ppf((1 + Q) / 2)
    m1 = int(np.floor(n * p - np.sqrt(n * p * (1 - p)) * z))
    m2 = int(np.ceil(n * p + np.sqrt(n * p * (1 - p)) * z))

    # Ограничение индексов в пределах размера выборки
    m1 = max(0, min(m1, n - 1))
    m2 = max(0, min(m2, n - 1))

    return m1, m2


# Функция для поиска минимального размера выборки, при котором доверительный интервал исключает X_(1) или X_(n)
def find_sample_size_exclusion(generate_data_func, distribution_name):
    quantiles = [0.01, 0.99]  # Квантили для анализа
    max_size = 10000  # Максимальный размер выборки
    step = 1  # Шаг увеличения размера выборки

    for q in quantiles:
        for N in range(step, max_size + 1, step):
            data = generate_data_func(N)
            sorted_data = np.sort(data)

            # Вычисление точного доверительного интервала
            m1_exact, m2_exact = exact_confidence_interval(N, p=q)
            ci_exact = (sorted_data[m1_exact], sorted_data[m2_exact])

            # Вычисление аппроксимированного доверительного интервала
            m1_approx, m2_approx = approximate_confidence_interval(N, p=q)
            ci_approx = (sorted_data[m1_approx], sorted_data[m2_approx])

            # Проверка условия исключения X_(1) или X_(n) для точного доверительного интервала
            if (q == 0.01 and m1_exact > 0) or (q == 0.99 and m2_exact < N - 1):
                print(f"{distribution_name} (Quantile {q}): Exact CI excludes X_(1) or X_(n) at sample size: {N}")
                break

            # Проверка условия исключения X_(1) или X_(n) для аппроксимированного доверительного интервала
            if (q == 0.01 and m1_approx > 0) or (q == 0.99 and m2_approx < N - 1):
                print(f"{distribution_name} (Quantile {q}): Approx CI excludes X_(1) or X_(n) at sample size: {N}")
                break
        print(f"{distribution_name} (Quantile {q}): Exact CI: {ci_exact}, Approx CI: {ci_approx} \n")


find_sample_size_exclusion(generate_uniform_minus1_1, "Uniform [-1, 1]")
find_sample_size_exclusion(generate_normal, "Normal (0, 1)")
find_sample_size_exclusion(generate_sum_2_uniform_minus1_1, "Sum of Uniforms")
