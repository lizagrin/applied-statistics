import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# Параметры задачи
mu = 0  # Истинное математическое ожидание
sigma = 1  # Истинное стандартное отклонение
Q = 0.95  # Доверительная вероятность
N = 10000  # Количество выборок для моделирования

# Размеры выборок для анализа
sample_sizes = [10, 20, 50, 100, 500]


# Функция для расчета D в доверительном интервале Хора-Хора
def D_rho(n, rho):
    log_term = np.log((1 - rho) / 4)
    return np.sqrt(-log_term / (2 * n)) - 1 / (6 * n)


# Сохранение доверительных интервалов для каждого n
confidence_intervals = {}

for n in sample_sizes:
    gamma_count = 0
    t_quantile = t.ppf(1 - (1 - Q) / 2, df=n - 1)  # Квантиль распределения Стьюдента
    D = D_rho(n, Q)  # Вычисляем D один раз для данного n

    gammas = []

    for _ in range(N):
        # Генерация выборки
        sample = np.random.normal(mu, sigma, n)

        # Выборочное среднее и стандартное отклонение
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)

        # Вариационный ряд
        sorted_sample = np.sort(sample)

        # Доверительные интервалы Хора-Хора
        hora_hora_contains = False
        for i in range(n//2, n // 2 + 1):
            a, b = sorted_sample[i - 1], sorted_sample[-i]
            hora_hora_lower = sample_mean - (b - a) * D
            hora_hora_upper = sample_mean + (b - a) * D

            # Проверяем, попадает ли mu в данный интервал Хора-Хора
            if hora_hora_lower <= mu <= hora_hora_upper:
                hora_hora_contains = True
                break

        # Доверительный интервал Стьюдента
        student_lower = sample_mean - t_quantile * (sample_std / np.sqrt(n))
        student_upper = sample_mean + t_quantile * (sample_std / np.sqrt(n))

        student_contains = student_lower <= mu <= student_upper

        if hora_hora_contains and not student_contains:
            gamma_count += 1

        # Сохраняем текущее значение lambda (доля gamma)
        gammas.append(gamma_count / (_ + 1))

    # Оценка доверительного интервала для lambda
    gamma_mean = np.mean(gammas)
    gamma_std = np.std(gammas, ddof=1)
    ci_lower = gamma_mean - t.ppf(1 - (1 - Q) / 2, df=N - 1) * gamma_std / np.sqrt(N)
    ci_upper = gamma_mean + t.ppf(1 - (1 - Q) / 2, df=N - 1) * gamma_std / np.sqrt(N)

    confidence_intervals[n] = (gamma_mean, ci_lower, ci_upper)

# Построение графика
plt.figure(figsize=(10, 6))
for n, (gamma_mean, ci_lower, ci_upper) in confidence_intervals.items():
    plt.errorbar(n, gamma_mean, yerr=[[gamma_mean - ci_lower], [ci_upper - gamma_mean]], fmt='o', label=f'n = {n}')

plt.xlabel("Размер выборки (n)")
plt.ylabel("Оценка доли gamma")
plt.title("Доверительный интервал для вероятности gamma")
plt.grid(True)
plt.legend()
plt.show()
