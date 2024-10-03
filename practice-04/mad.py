import numpy as np
from scipy.stats import norm


def generate_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size)


def generate_normal(size):
    return np.random.normal(0, 1, size)


def generate_sum_2_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size) + np.random.uniform(-1, 1, size)


def mad(data):
    # Ищем среднее абсолютное отклонение (MAD)
    median = np.median(data)
    return np.mean(np.abs(data - median))


def mad_confidence_interval(data, confidence_level=0.95):
    # Ищем доверительный интервал для MAD
    n = len(data)
    mad_value = mad(data)

    # Оценка стандартного отклонения MAD
    # Приближенная оценка: σ = MAD / 0.67449 (для нормального распределения)
    sigma_est = mad_value / 0.67449

    # Z-значение для заданного уровня доверия
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)

    # Полуширина доверительного интервала
    half_width = z_value * sigma_est / np.sqrt(n)

    # Доверительный интервал
    ci_lower = mad_value - half_width
    ci_upper = mad_value + half_width

    return ci_lower, ci_upper


# Размер выборки
sample_size = 10000

# Генерация данных и вычисление доверительных интервалов
data_uniform = generate_uniform_minus1_1(sample_size)
mad_uniform = mad(data_uniform)
ci_uniform = mad_confidence_interval(data_uniform)

data_normal = generate_normal(sample_size)
mad_normal = mad(data_normal)
ci_normal = mad_confidence_interval(data_normal)

data_sum_uniform = generate_sum_2_uniform_minus1_1(sample_size)
mad_sum_uniform = mad(data_sum_uniform)
ci_sum_uniform = mad_confidence_interval(data_sum_uniform)

print(f"Uniform (-1, 1): MAD = {mad_uniform:.4f}, 95% CI = ({ci_uniform[0]:.4f}, {ci_uniform[1]:.4f})")
print(f"Normal (0, 1): MAD = {mad_normal:.4f}, 95% CI = ({ci_normal[0]:.4f}, {ci_normal[1]:.4f})")
print(
    f"Sum of two Uniform (-1, 1): MAD = {mad_sum_uniform:.4f}, 95% CI = ({ci_sum_uniform[0]:.4f}, {ci_sum_uniform[1]:.4f})")
