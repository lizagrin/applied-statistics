import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Генераторы выборок
def generate_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N)


def generate_normal(N):
    return np.random.normal(0, 1, N)


def generate_sum_2_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N) + np.random.uniform(-1, 1, N)


# Формула (1) для вычисления доверительных интервалов
def confidence_interval(sample, confidence=0.95):
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample, ddof=1)

    # Квантиль для стандартного нормального распределения
    z_score = norm.ppf((1 + confidence) / 2)

    # Вычисление границ доверительного интервала
    margin_of_error = z_score * (np.sqrt(sample_variance) / np.sqrt(n))
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return lower_bound, upper_bound


# TODO
# вот здесь считаем долю доверительных интервалов для формул (2) и (3) и выводим графики

# Параметры задачи
n_values = [10, 100, 1000, 5000, 10000]
n_trials = 1000
p = 0.95

# Для хранения результатов
generators = {
    'Uniform': generate_uniform_minus1_1,
    'Normal': generate_normal,
    'Sum of uniform': generate_sum_2_uniform_minus1_1
}

# Генерация выборок и расчет доверительных интервалов
intervals = {gen: [] for gen in generators}

for gen_name, gen_func in generators.items():
    for N in n_values:
        lower_bounds = []
        upper_bounds = []
        for _ in range(n_trials):
            sample = gen_func(N)
            ci = confidence_interval(sample, confidence=p)
            lower_bounds.append(ci[0])
            upper_bounds.append(ci[1])
        intervals[gen_name].append((np.mean(lower_bounds), np.mean(upper_bounds)))

# Визуализация доверительных интервалов
plt.figure(figsize=(10, 6))

for gen_name in generators:
    lower_bounds, upper_bounds = zip(*intervals[gen_name])
    plt.plot(n_values, lower_bounds, marker='o', linestyle='-', label=f'{gen_name} lower bound')
    plt.plot(n_values, upper_bounds, marker='o', linestyle='-', label=f'{gen_name} upper bound')

plt.xscale('log')
plt.xlabel('Размер выборки (n)')
plt.ylabel('Границы доверительного интервала')
plt.title('Доверительные интервалы для различных распределений')
plt.grid(True, linewidth=0.3)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
