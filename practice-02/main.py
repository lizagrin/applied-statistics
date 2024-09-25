import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm


# Генераторы выборок
def generate_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N)


def generate_normal(N):
    return np.random.normal(0, 1, N)


def generate_sum_2_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N) + np.random.uniform(-1, 1, N)


# Функция для вычисления доверительных интервалов
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


# Функция для генерации выборок и вычисления средних значений
def generate_samples_and_means(n_values):
    means_uniform = []
    means_normal = []
    means_cauchy = []

    for n in n_values:  # Проходит по каждому значению n из списка n_values
        uniform_samples = generate_uniform_minus1_1(n)
        means_uniform.append(np.mean(uniform_samples))

        normal_samples = generate_normal(n)
        means_normal.append(np.mean(normal_samples))

        cauchy_samples = np.random.standard_cauchy(size=n)
        means_cauchy.append(np.mean(cauchy_samples))

    return means_uniform, means_normal, means_cauchy  # Возвращает списки средних значений


# Функция для визуализации средних значений
def plot_means(n_values, means_uniform, means_normal, means_cauchy):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(n_values, means_uniform, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.title('Среднее для равномерного распределения')
    plt.xlabel('n')
    plt.ylabel('Среднее значение')

    plt.subplot(1, 3, 2)
    plt.plot(n_values, means_normal, marker='o', linestyle='-', color='g')
    plt.xscale('log')
    plt.title('Среднее для нормального распределения')
    plt.xlabel('n')
    plt.ylabel('Среднее значение')

    plt.subplot(1, 3, 3)
    plt.plot(n_values, means_cauchy, marker='o', linestyle='-', color='r')
    plt.xscale('log')
    plt.title('Среднее для распределения Коши')
    plt.xlabel('n')
    plt.ylabel('Среднее значение')

    plt.tight_layout()
    plt.show()


# Параметры задачи
n_values = [10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
n_values_2 = [10, 100, 1000, 5000]
n_trials = 10000
p = 0.95
true_mean = 0

# Для хранения результатов
generators = {
    'uniform_minus1_1': generate_uniform_minus1_1,
    'normal': generate_normal,
    'sum_2_uniform_minus1_1': generate_sum_2_uniform_minus1_1
}  # Словарь генераторов выборок

interval_coverage = {gen: [] for gen in
                     generators}  # Словарь для хранения долей интервалов, содержащих истинное среднее

# Генерация выборок, расчет доверительных интервалов и проверка попадания 0 в интервалы
for gen_name, gen_func in generators.items():
    for N in n_values_2:
        count_included_zero = 0
        for _ in range(n_trials):
            sample = gen_func(N)
            ci = confidence_interval(sample, confidence=p)
            if ci[0] <= true_mean <= ci[1]:
                count_included_zero += 1
        interval_coverage[gen_name].append(count_included_zero / n_trials)

# Вывод результатов
for gen_name in generators:  # Проходит по каждому генератору в словаре generators
    print(f"\nГенератор: {gen_name}")
    for i, N in enumerate(n_values_2):
        print(f"N={N}, Доля интервалов, содержащих 0: {interval_coverage[gen_name][i]:.4f}")

# Визуализация доверительных интервалов
plt.figure(figsize=(10, 6))
for gen_name in interval_coverage:
    plt.plot(n_values_2, interval_coverage[gen_name], label=gen_name)

plt.xscale('log')
plt.xlabel('Размер выборки (log scale)')
plt.ylabel('Доля интервалов, содержащих 0')
plt.title(f'Доверительные интервалы (p = {p}) для разных генераторов')

plt.legend()
plt.grid(True)
plt.show()

# Генерация выборок и вычисление средних значений
means_uniform, means_normal, means_cauchy = generate_samples_and_means(n_values)

# Визуализация средних значений
plot_means(n_values, means_uniform, means_normal, means_cauchy)
