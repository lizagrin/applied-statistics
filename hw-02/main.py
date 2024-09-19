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


# Функция для проведения эксперимента и оценки покрытия доверительных интервалов
def evaluate_coverage(distribution_func):
    coverage_results_bootstrap = []  # Списки для хранения результатов покрытия
    coverage_results_jackknife = []

    for size in sample_sizes:
        bootstrap_coverage_count = 0  # Счетчики для успешных покрытий
        jackknife_coverage_count = 0

        for _ in range(num_samples):
            data = distribution_func(size)  # Генерируем случайную выборку заданного размера

            lower_bootstrap, upper_bootstrap = bootstrap(data, alpha=alpha)
            if lower_bootstrap <= true_mean <= upper_bootstrap:
                bootstrap_coverage_count += 1  # Увеличиваем счетчик, если истинное значение покрыто

            lower_jackknife, upper_jackknife = jackknife(data)
            if lower_jackknife <= true_mean <= upper_jackknife:
                jackknife_coverage_count += 1  # Увеличиваем счетчик, если истинное значение покрыто

        # Оценка доли покрытий для текущего размера выборки
        bootstrap_coverage_rate = bootstrap_coverage_count / num_samples
        jackknife_coverage_rate = jackknife_coverage_count / num_samples

        coverage_results_bootstrap.append(bootstrap_coverage_rate)  # Добавляем результаты в списки
        coverage_results_jackknife.append(jackknife_coverage_rate)

    return coverage_results_bootstrap, coverage_results_jackknife  # Возвращаем результаты покрытия


# Генераторы случайных выборок для разных распределений
uniform_dist = lambda size: np.random.uniform(-1, 1, size)
normal_dist = lambda size: np.random.normal(0, 1, size)
sum_uniform_dist = lambda size: np.random.uniform(-1, 1, size) + np.random.uniform(-1, 1, size)

# Оценка покрытия для каждого распределения
uniform_bootstrap, uniform_jackknife = evaluate_coverage(uniform_dist)
normal_bootstrap, normal_jackknife = evaluate_coverage(normal_dist)
sum_uniform_bootstrap, sum_uniform_jackknife = evaluate_coverage(sum_uniform_dist)

# Построение графиков
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(sample_sizes, uniform_bootstrap, label='Равномерное распределение', marker='o')
plt.plot(sample_sizes, normal_bootstrap, label='Нормальное распределение', marker='o')
plt.plot(sample_sizes, sum_uniform_bootstrap, label='Сумма равномерных', marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='Ожидаемое покрытие (0.95)')
plt.xlabel('Размер выборки')
plt.ylabel('Вероятность покрытия')
plt.title('Bootstrap')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(sample_sizes, uniform_jackknife, label='Равномерное распределение', marker='o')
plt.plot(sample_sizes, normal_jackknife, label='Нормальное распределение', marker='o')
plt.plot(sample_sizes, sum_uniform_jackknife, label='Сумма равномерных', marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='Ожидаемое покрытие (0.95)')
plt.xlabel('Размер выборки')
plt.ylabel('Вероятность покрытия')
plt.title('Jackknife')
plt.legend()

plt.tight_layout()
plt.show()
