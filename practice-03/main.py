import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

plt.rcParams.update({'font.size': 18})


# Генераторы выборок
def generate_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N)


def generate_normal(N):
    return np.random.normal(0, 1, N)


def generate_sum_2_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N) + np.random.uniform(-1, 1, N)

# Формула (1) для вычисления доверительных интервалов с использованием нормального распределения
def confidence_interval(sample, confidence=0.95):
    n = len(sample)  # Размер выборки
    sample_mean = np.mean(sample)  # Среднее значение выборки
    sample_variance = np.var(sample, ddof=1)  # Выборочная дисперсия

    # Квантиль для стандартного нормального распределения
    z_score = norm.ppf((1 + confidence) / 2)

    # Вычисление границ доверительного интервала
    margin_of_error = z_score * (np.sqrt(sample_variance) / np.sqrt(n))  # Погрешность
    lower_bound = sample_mean - margin_of_error  # Нижняя граница интервала
    upper_bound = sample_mean + margin_of_error  # Верхняя граница интервала
    return lower_bound, upper_bound


# Формула (2) для вычисления доверительных интервалов с использованием распределения Стьюдента
def confidence_interval_student(sample, confidence=0.95):
    n = len(sample)  # Размер выборки
    sample_mean = np.mean(sample)  # Среднее значение выборки
    sample_std = np.std(sample, ddof=1)  # Стандартное отклонение выборки

    # Квантиль для распределения Стьюдента с n-1 степенями свободы
    t_score = t.ppf((1 + confidence) / 2, n - 1)

    # Вычисление границ доверительного интервала
    margin_of_error = t_score * (sample_std / np.sqrt(n))  # Погрешность
    lower_bound = sample_mean - margin_of_error  # Нижняя граница интервала
    upper_bound = sample_mean + margin_of_error  # Верхняя граница интервала
    return lower_bound, upper_bound


# Формула (3) для вычисления доверительных интервалов на основе эмпирического распределения
def confidence_interval_empirical(sample, confidence=0.95):
    sorted_sample = np.sort(sample)  # Сортировка выборки по возрастанию
    a = sorted_sample[0]  # Минимальное значение в выборке
    b = sorted_sample[-1]  # Максимальное значение в выборке

    n = len(sample)  # Размер выборки
    sample_mean = np.mean(sample)  # Среднее значение выборки

    # Эмпирическая формула для вычисления погрешности
    D = np.sqrt(-np.log((1 - confidence) / 4) / (2 * n)) - 1 / (6 * n)
    margin_of_error = (b - a) * D  # Погрешность

    # Вычисление границ доверительного интервала
    lower_bound = sample_mean - margin_of_error  # Нижняя граница интервала
    upper_bound = sample_mean + margin_of_error  # Верхняя граница интервала
    return lower_bound, upper_bound


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

# Цвета для каждой границы распределения
colors = {
    'Uniform_lower': 'blue',
    'Uniform_upper': 'cyan',
    'Normal_lower': 'green',
    'Normal_upper': 'lime',
    'Sum of uniform_lower': 'red',
    'Sum of uniform_upper': 'orange'
}

# Истинные математические ожидания для каждого распределения
true_means = {
    'Uniform': 0,
    'Normal': 0,
    'Sum of uniform': 0
}

# Генерация выборок и расчет доверительных интервалов
intervals_formula_1 = {gen: [] for gen in generators}
intervals_formula_2 = {gen: [] for gen in generators}
intervals_formula_3 = {gen: [] for gen in generators}

coverage_formula_1 = {gen: [] for gen in generators}
coverage_formula_2 = {gen: [] for gen in generators}
coverage_formula_3 = {gen: [] for gen in generators}

for gen_name, gen_func in generators.items():
    true_mean = true_means[gen_name]

    for N in n_values:
        lower_bounds_1 = []
        upper_bounds_1 = []
        lower_bounds_2 = []
        upper_bounds_2 = []
        lower_bounds_3 = []
        upper_bounds_3 = []

        coverage_count_1 = 0
        coverage_count_2 = 0
        coverage_count_3 = 0

        for _ in range(n_trials):
            sample = gen_func(N)
            ci_1 = confidence_interval(sample, confidence=p)
            ci_2 = confidence_interval_student(sample, confidence=p)
            ci_3 = confidence_interval_empirical(sample, confidence=p)

            lower_bounds_1.append(ci_1[0])
            upper_bounds_1.append(ci_1[1])
            lower_bounds_2.append(ci_2[0])
            upper_bounds_2.append(ci_2[1])
            lower_bounds_3.append(ci_3[0])
            upper_bounds_3.append(ci_3[1])

            # Подсчет покрытия
            if ci_1[0] <= true_mean <= ci_1[1]:
                coverage_count_1 += 1
            if ci_2[0] <= true_mean <= ci_2[1]:
                coverage_count_2 += 1
            if ci_3[0] <= true_mean <= ci_3[1]:
                coverage_count_3 += 1

        # Добавление среднего значения нижней и верхней границ доверительных интервалов
        intervals_formula_1[gen_name].append((np.mean(lower_bounds_1), np.mean(upper_bounds_1)))
        intervals_formula_2[gen_name].append((np.mean(lower_bounds_2), np.mean(upper_bounds_2)))
        intervals_formula_3[gen_name].append((np.mean(lower_bounds_3), np.mean(upper_bounds_3)))

        # Сохранение доли случаев, когда истинное значение попадает в доверительный интервал
        coverage_formula_1[gen_name].append(coverage_count_1 / n_trials)
        coverage_formula_2[gen_name].append(coverage_count_2 / n_trials)
        coverage_formula_3[gen_name].append(coverage_count_3 / n_trials)

# Визуализация доверительных интервалов для всех трех формул на одном графике
plt.figure(figsize=(32, 20))

for gen_name in generators:
    # Формула 1
    lower_bounds_1, upper_bounds_1 = zip(*intervals_formula_1[gen_name])
    plt.plot(n_values, lower_bounds_1, marker='o', linestyle='-', color=colors[f'{gen_name}_lower'],
             label=f'{gen_name} lower bound (Formula 1)', alpha=0.7)
    plt.plot(n_values, upper_bounds_1, marker='o', linestyle='-', color=colors[f'{gen_name}_upper'],
             label=f'{gen_name} upper bound (Formula 1)', alpha=0.7)

    # Формула 2
    lower_bounds_2, upper_bounds_2 = zip(*intervals_formula_2[gen_name])
    plt.plot(n_values, lower_bounds_2, marker='x', linestyle='--', color=colors[f'{gen_name}_lower'],
             label=f'{gen_name} lower bound (Formula 2)', alpha=0.7)
    plt.plot(n_values, upper_bounds_2, marker='x', linestyle='--', color=colors[f'{gen_name}_upper'],
             label=f'{gen_name} upper bound (Formula 2)', alpha=0.7)

    # Формула 3
    lower_bounds_3, upper_bounds_3 = zip(*intervals_formula_3[gen_name])
    plt.plot(n_values, lower_bounds_3, marker='s', linestyle='-.', color=colors[f'{gen_name}_lower'],
             label=f'{gen_name} lower bound (Formula 3)', alpha=0.7)
    plt.plot(n_values, upper_bounds_3, marker='s', linestyle='-.', color=colors[f'{gen_name}_upper'],
             label=f'{gen_name} upper bound (Formula 3)', alpha=0.7)

plt.xscale('log')
plt.xlabel('Sample size (n)', fontsize=30)
plt.ylabel('Confidence interval boundaries', fontsize=30)
plt.title('Confidence intervals for different distributions', fontsize=30)
plt.grid(True, linewidth=0.3)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# Вывод доли покрытия для каждого генератора и формулы
for gen_name in generators:
    print(f"{gen_name}:")
    print(f"  Formula 1 coverage: {coverage_formula_1[gen_name]}")
    print(f"  Formula 2 coverage: {coverage_formula_2[gen_name]}")
    print(f"  Formula 3 coverage: {coverage_formula_3[gen_name]} \n")

