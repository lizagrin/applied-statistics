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
    sorted_sample = np.sort(sample)
    a = sorted_sample[1]  # Второе минимальное значение в выборке
    b = sorted_sample[-2]  # Второе максимальное значение в выборке

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
n_values = [50, 100, 500, 1000, 5000, 10000]
n_trials = 1000
p = 0.95

# Для хранения результатов
generators = {
    'Uniform': generate_uniform_minus1_1,
    'Normal': generate_normal,
    'Sum of uniform': generate_sum_2_uniform_minus1_1
}

colors = {
    'Uniform_lower': 'blue',
    'Uniform_upper': 'cyan',
    'Normal_lower': 'green',
    'Normal_upper': 'lime',
    'Sum of uniform_lower': 'red',
    'Sum of uniform_upper': 'orange'
}

true_means = {
    'Uniform': 0,
    'Normal': 0,
    'Sum of uniform': 0
}

intervals_formula_1 = {gen: [] for gen in generators}
intervals_formula_2 = {gen: [] for gen in generators}
intervals_formula_3 = {gen: [] for gen in generators}

# Для каждого генератора выборок (равномерного, нормального и суммы двух равномерных)
for gen_name, gen_func in generators.items():
    # Истинное среднее значение для текущего распределения
    true_mean = true_means[gen_name]

    # Для каждого значения размера выборки
    for N in n_values:
        # Списки для хранения нижних и верхних границ доверительных интервалов для каждой из трех формул
        lower_bounds_1 = []
        upper_bounds_1 = []
        lower_bounds_2 = []
        upper_bounds_2 = []
        lower_bounds_3 = []
        upper_bounds_3 = []

        # Проведение нескольких экспериментов для получения статистически значимых результатов
        for _ in range(n_trials):
            # Генерация выборки заданного размера N
            sample = gen_func(N)
            # Добавление выброса 100 в выборку
            sample_with_outlier = np.append(sample, 100)

            # Вычисление доверительных интервалов для выборки с выбросом по трем различным методам
            lb1, ub1 = confidence_interval(sample_with_outlier, p)
            lb2, ub2 = confidence_interval_student(sample_with_outlier, p)
            lb3, ub3 = confidence_interval_empirical(sample_with_outlier, p)

            # Сохранение нижних и верхних границ для каждого метода
            lower_bounds_1.append(lb1)
            upper_bounds_1.append(ub1)
            lower_bounds_2.append(lb2)
            upper_bounds_2.append(ub2)
            lower_bounds_3.append(lb3)
            upper_bounds_3.append(ub3)

        # Средние значения нижних и верхних границ доверительных интервалов для текущего размера выборки
        intervals_formula_1[gen_name].append((np.mean(lower_bounds_1), np.mean(upper_bounds_1)))
        intervals_formula_2[gen_name].append((np.mean(lower_bounds_2), np.mean(upper_bounds_2)))
        intervals_formula_3[gen_name].append((np.mean(lower_bounds_3), np.mean(upper_bounds_3)))


# Функция для построения графиков
def plot_intervals(n_values, intervals_dict, title):
    plt.figure(figsize=(14, 8))

    for gen_name in intervals_dict:
        lower_bounds, upper_bounds = zip(*intervals_dict[gen_name])
        plt.plot(n_values, lower_bounds, label=f'{gen_name} Lower', color=colors[f'{gen_name}_lower'], linestyle='-',
                 marker='o')
        plt.plot(n_values, upper_bounds, label=f'{gen_name} Upper', color=colors[f'{gen_name}_upper'], linestyle='--',
                 marker='o')

    plt.xlabel('Sample Size')
    plt.ylabel('Confidence Interval Bound')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Построение графиков
plot_intervals(n_values, intervals_formula_1, 'Confidence Intervals using Normal Distribution (formula 1)')
plot_intervals(n_values, intervals_formula_2, 'Confidence Intervals using Student\'s t-Distribution (formula 2)')
plot_intervals(n_values, intervals_formula_3, 'Confidence Intervals using Empirical Distribution (formula 3)')
