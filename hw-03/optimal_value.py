import numpy as np


# Генераторы выборок
def generate_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N)


def generate_normal(N):
    return np.random.normal(0, 1, N)


def generate_sum_2_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N) + np.random.uniform(-1, 1, N)


# Формула (3) для вычисления доверительных интервалов на основе эмпирического распределения
def confidence_interval_empirical(sample, confidence=0.95, k=0.025):
    sorted_sample = np.sort(sample)
    n = len(sample)

    # Определение индексов для a и b
    k_n = int(np.floor(k * n))
    n_k_n = int(np.ceil((1 - k) * n)) - 1

    a = sorted_sample[k_n]
    b = sorted_sample[n_k_n]

    sample_mean = np.mean(sample)

    # Эмпирическая формула для вычисления погрешности
    D = np.sqrt(-np.log((1 - confidence) / 4) / (2 * n)) - 1 / (6 * n)
    margin_of_error = (b - a) * D

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    return lower_bound, upper_bound


# Параметры задачи
n_values = [10, 100, 1000, 5000]
n_trials = 1000
p = 0.95

# Для хранения результатов
generators = {
    'Uniform': generate_uniform_minus1_1,
    'Normal': generate_normal,
    'Sum of uniform': generate_sum_2_uniform_minus1_1
}

# Истинные математические ожидания для каждого распределения
true_means = {
    'Uniform': 0,
    'Normal': 0,
    'Sum of uniform': 0
}

# Поиск оптимального k
optimal_k_values = {}

# Перебираем каждый генератор выборок в словаре generators
for gen_name, gen_func in generators.items():
    # Получаем истинное математическое ожидание для текущего распределения
    true_mean = true_means[gen_name]
    # Инициализируем переменные для хранения оптимального значения k и минимальной разницы
    optimal_k = None
    min_diff = float('inf')
    optimal_coverage_probability = 0  # Переменная для хранения вероятности покрытия для оптимального k

    # Исследуем разные значения k от 0.001 до 0.1 с шагом, чтобы найти оптимальное
    for k in np.linspace(0.1, 0.9, 100):
        coverage_count = 0  # Счетчик для подсчета количества попаданий истинного среднего в доверительный интервал

        # Перебираем различные размеры выборок из списка n_values
        for N in n_values:
            # Проводим n_trials экспериментов для текущего размера выборки и значения k
            for _ in range(n_trials):
                # Генерируем выборку с использованием текущего генератора
                sample = gen_func(N)
                # Вычисляем доверительный интервал для текущей выборки и значения k
                ci_3 = confidence_interval_empirical(sample, confidence=p, k=k)

                # Проверяем, попадает ли истинное среднее в доверительный интервал
                if ci_3[0] <= true_mean <= ci_3[1]:
                    coverage_count += 1  # Увеличиваем счетчик попаданий

        # Вычисляем фактическую вероятность покрытия (сколько раз истинное среднее попало в доверительный интервал)
        actual_coverage_probability = coverage_count / (n_trials * len(n_values))

        # Вычисляем разницу между фактической вероятностью покрытия и заданной вероятностью p
        diff = abs(actual_coverage_probability - p)
        # Если текущая разница < минимальной найденной ранее, обновляем минимальную разницу и оптимальное значение k
        if diff < min_diff:
            min_diff = diff
            optimal_k = k
            optimal_coverage_probability = actual_coverage_probability  # Сохраняем вероятность для оптимального k

    # Сохраняем найденное оптимальное значение k для текущего генератора выборок
    optimal_k_values[gen_name] = optimal_k

    # Выводим фактическую вероятность покрытия для оптимального значения k
    print(
        f"Generator: {gen_name}, Optimal k: {optimal_k:.3f}, Actual coverage probability: {optimal_coverage_probability:.4f}")

