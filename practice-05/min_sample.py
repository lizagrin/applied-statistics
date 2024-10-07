import numpy as np
from scipy.stats import norm

# Квантиль для нормального распределения при Q = 0.95
Q = 0.95
z = norm.ppf((1 + Q) / 2)

# Допустимые погрешности Δ_доп
deltas = [10 ** -i for i in range(1, 4)]  # 10^-1, 10^-2, 10^-3

# Распределения
distributions = {
    'Uniform(-1, 1)': lambda n: np.random.uniform(-1, 1, n),
    'Normal(0, 1)': lambda n: np.random.normal(0, 1, n),
    'Sum of two Uniform(-1, 1)': lambda n: np.random.uniform(-1, 1, n) + np.random.uniform(-1, 1, n)
}

# Дисперсии для распределений
variances = {
    'Uniform(-1, 1)': 1 / 3,
    'Normal(0, 1)': 1,
    'Sum of two Uniform(-1, 1)': 2 / 3
}


# Функция для вычисления необходимого n
def required_sample_size(variance, delta):
    return (variance * (z ** 2)) / (delta ** 2)


for dist_name, dist_func in distributions.items():
    print(f"Распределение: {dist_name}")
    variance = variances[dist_name]

    for delta in deltas:
        # Вычисляем минимальный размер выборки
        n_min = int(np.ceil(required_sample_size(variance, delta)))
        print(f"  Для Δ_доп = {delta}, минимальный n = {n_min}")

        # Генерируем выборку минимального размера
        sample = dist_func(n_min)

        # Оценка математического ожидания
        mean_estimate = np.mean(sample)
        print(f"  Оценка математического ожидания: {mean_estimate:.6f}")
    print()
