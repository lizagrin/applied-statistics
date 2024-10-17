import numpy as np
from scipy.stats import norm

n = 10000

# Генерация распределений
uniform_dist = np.random.uniform(-1, 1, n)
normal_dist = np.random.normal(0, 1, n)
combined_dist = np.random.uniform(-1, 1, n) + np.random.uniform(-1, 1, n)

# Рассчет 1% и 95% квантилей
quantiles_uniform = np.percentile(uniform_dist, [1, 95])
quantiles_normal = np.percentile(normal_dist, [1, 95])
quantiles_combined = np.percentile(combined_dist, [1, 95])

print(f"Квантили для равномерного распределения [-1, 1]: {quantiles_uniform}")
print(f"Квантили для нормального распределения N(0, 1): {quantiles_normal}")
print(f"Квантили для суммы двух равномерных распределений: {quantiles_combined} \n")


# Расчет размеров выборок для доверительных интервалов
def sample_size_for_confidence_interval(Q, p):
    # Значение z для доверительного уровня Q
    z = norm.ppf((1 + Q) / 2)

    # Формула для k1 (левая граница)
    def k1_formula(p, z):
        return (np.sqrt(1 - p) * z + np.sqrt((1 - p) * z ** 2 + 4)) / (2 * np.sqrt(p))

    # Формула для k2 (правая граница)
    def k2_formula(p, z):
        return (np.sqrt(p) * z + np.sqrt(p * z ** 2 + 4)) / (2 * np.sqrt(1 - p))

    # Определяем минимальные n для каждой из границ
    k1 = k1_formula(p, z)
    k2 = k2_formula(p, z)

    n1 = np.ceil(k1 ** 2)  # Левая граница (квадрат корня)
    n2 = np.ceil(k2 ** 2)  # Правая граница

    return max(n1, n2)


# Рассчитаем для Q = 0.90, 0.95, 0.99
sample_sizes_01 = {
    0.90: sample_size_for_confidence_interval(0.90, 0.01),
    0.95: sample_size_for_confidence_interval(0.95, 0.01),
    0.99: sample_size_for_confidence_interval(0.99, 0.01),
}
sample_sizes_95 = {
    0.90: sample_size_for_confidence_interval(0.90, 0.95),
    0.95: sample_size_for_confidence_interval(0.95, 0.95),
    0.99: sample_size_for_confidence_interval(0.99, 0.95),
}
print(f"Необходимые размеры выборок для доверительных интервалов для p = 0.01: {sample_sizes_01}")
print(f"Необходимые размеры выборок для доверительных интервалов для p = 0.95: {sample_sizes_95} \n")


# Генерация выборок и проверка доверительных интервалов для всех распределений
def check_confidence_intervals(sample_size, dist):
    # Генерация указанного распределения
    sample = dist(int(sample_size))

    # Доверительные интервалы для 1% и 95% квантилей
    lower_bound, upper_bound = np.percentile(sample, [1, 95])

    # Проверка: попадают ли границы в диапазон выборки
    min_val, max_val = np.min(sample), np.max(sample)

    return lower_bound >= min_val and upper_bound <= max_val


# Проверим для всех размеров выборок и всех распределений
distributions = {
    "uniform": lambda n: np.random.uniform(-1, 1, n),
    "normal": lambda n: np.random.normal(0, 1, n),
    "sum of uniform": lambda n: np.random.uniform(-1, 1, n) + np.random.uniform(-1, 1, n)
}

results = {}
for name, dist in distributions.items():
    interval_checks_01 = {Q: check_confidence_intervals(n, dist) for Q, n in sample_sizes_01.items()}
    interval_checks_95 = {Q: check_confidence_intervals(n, dist) for Q, n in sample_sizes_95.items()}
    results[name] = {
        "p = 0.01": interval_checks_01,
        "p = 0.95": interval_checks_95
    }

# Вывод результатов
for dist_name, check_results in results.items():
    print(f"Результаты проверки для распределения {dist_name}:")
    for p_value, res in check_results.items():
        print(f"  Для {p_value}: {res}")
    print()
