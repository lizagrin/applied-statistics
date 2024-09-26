import matplotlib.pyplot as plt
import numpy as np


def bootstrap(data, num_bootstrap=500, alpha=0.05):
    n = len(data)  # Определяем размер выборки
    means = np.empty(num_bootstrap)  # Создаем массив для хранения средних значений выборок

    # Генерация выборок
    for i in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        # Вычисляем среднее значение выборки
        means[i] = np.mean(sample)

    # Вычисляем границы доверительного интервала
    lower_bound = np.percentile(means, 100 * (alpha / 2))
    upper_bound = np.percentile(means, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound


def jackknife(data):
    n = len(data)  # Определяем размер выборки
    means = np.empty(n)  # Создаем массив для хранения средних значений

    # Генерация выборок
    for i in range(n):
        # Удаляем i-й элемент из выборки, формируя выборку
        jackknife_sample = np.delete(data, i)
        # Вычисляем среднее значение
        means[i] = np.mean(jackknife_sample)

    # Вычисляем среднее значение и стандартную ошибку джекнайфа
    mean_jackknife = np.mean(means)
    se_jackknife = np.sqrt((n - 1) * np.var(means, ddof=1))

    # Вычисляем границы доверительного интервала на основе нормального распределения
    lower_bound = mean_jackknife - 1.96 * se_jackknife
    upper_bound = mean_jackknife + 1.96 * se_jackknife
    return lower_bound, upper_bound


# Параметры задачи (уменьшила для ускорения работы)
sample_sizes = [10, 100, 500, 1000, 2000]  # Различные размеры выборок
num_samples = 1000  # Количество повторений
alpha = 0.05  # Уровень значимости для доверительных интервалов
true_mean = 0  # Истинное математическое ожидание для проверки покрытия


def evaluate_intervals(distribution_func):
    # Списки для хранения средних доверительных интервалов для каждого размера выборки
    intervals_bootstrap = []
    intervals_jackknife = []

    # Перебираем каждый размер выборки из списка sample_sizes
    for size in sample_sizes:
        # Списки для хранения доверительных интервалов для каждой выборки текущего размера
        bootstrap_intervals = []
        jackknife_intervals = []

        # Генерируем num_samples выборок и вычисляем доверительные интервалы для каждой
        for _ in range(num_samples):
            # Генерация случайной выборки заданного размера с использованием переданной функции распределения
            data = distribution_func(size)

            # Вычисление доверительного интервала методом бутстрапа
            lower_bootstrap, upper_bootstrap = bootstrap(data, alpha=alpha)
            bootstrap_intervals.append((lower_bootstrap, upper_bootstrap))

            # Вычисление доверительного интервала методом джекнайфа
            lower_jackknife, upper_jackknife = jackknife(data)
            jackknife_intervals.append((lower_jackknife, upper_jackknife))

        # Вычисление среднего доверительного интервала для бутстрапа по всем сгенерированным выборкам
        avg_bootstrap_interval = (
            np.mean([interval[0] for interval in bootstrap_intervals]),  # Среднее нижних границ
            np.mean([interval[1] for interval in bootstrap_intervals])  # Среднее верхних границ
        )

        # Вычисление среднего доверительного интервала для джекнайфа по всем сгенерированным выборкам
        avg_jackknife_interval = (
            np.mean([interval[0] for interval in jackknife_intervals]),  # Среднее нижних границ
            np.mean([interval[1] for interval in jackknife_intervals])  # Среднее верхних границ
        )

        # Добавление среднего доверительного интервала для текущего размера выборки в общий список
        intervals_bootstrap.append(avg_bootstrap_interval)
        intervals_jackknife.append(avg_jackknife_interval)

    # Возвращаем списки средних доверительных интервалов для каждого метода
    return intervals_bootstrap, intervals_jackknife


# Генераторы случайных выборок
uniform_dist = lambda size: np.random.uniform(-1, 1, size)
normal_dist = lambda size: np.random.normal(0, 1, size)
sum_uniform_dist = lambda size: np.random.uniform(-1, 1, size) + np.random.uniform(-1, 1, size)

# Оценка интервалов для каждого распределения
uniform_intervals_bootstrap, uniform_intervals_jackknife = evaluate_intervals(uniform_dist)
normal_intervals_bootstrap, normal_intervals_jackknife = evaluate_intervals(normal_dist)
sum_uniform_intervals_bootstrap, sum_uniform_intervals_jackknife = evaluate_intervals(sum_uniform_dist)


# Построение графиков
def plot_intervals(sample_sizes, intervals_bootstrap, intervals_jackknife, title):
    plt.figure(figsize=(12, 6))

    # Границы доверительных интервалов для Bootstrap
    lower_bounds_bootstrap = [interval[0] for interval in intervals_bootstrap]
    upper_bounds_bootstrap = [interval[1] for interval in intervals_bootstrap]

    # Границы доверительных интервалов для Jackknife
    lower_bounds_jackknife = [interval[0] for interval in intervals_jackknife]
    upper_bounds_jackknife = [interval[1] for interval in intervals_jackknife]

    plt.plot(sample_sizes, lower_bounds_bootstrap, label='Bootstrap Lower Bound', marker='o')
    plt.plot(sample_sizes, upper_bounds_bootstrap, label='Bootstrap Upper Bound', marker='o')

    plt.plot(sample_sizes, lower_bounds_jackknife, label='Jackknife Lower Bound', marker='x')
    plt.plot(sample_sizes, upper_bounds_jackknife, label='Jackknife Upper Bound', marker='x')

    plt.xlabel('Sample Size')
    plt.ylabel('Confidence Interval Bound')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_intervals(sample_sizes, uniform_intervals_bootstrap, uniform_intervals_jackknife, 'Uniform Distribution')
plot_intervals(sample_sizes, normal_intervals_bootstrap, normal_intervals_jackknife, 'Normal Distribution')
plot_intervals(sample_sizes, sum_uniform_intervals_bootstrap, sum_uniform_intervals_jackknife,
               'Sum of Uniform Distributions')
