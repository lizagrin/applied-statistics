import numpy as np
from scipy.stats import chi2, cauchy
from scipy.optimize import minimize


# Функция для вычисления доверительных интервалов для дисперсии
def confidence_interval_variance(x, alpha=0.05):
    n = len(x)
    var = np.var(x, ddof=1)
    chi2_left = chi2.ppf(alpha / 2, n - 1)
    chi2_right = chi2.ppf(1 - alpha / 2, n - 1)
    lower_bound = (n - 1) * var / chi2_right
    upper_bound = (n - 1) * var / chi2_left
    return lower_bound, upper_bound


# Генерация случайных выборок по распределению Коши
def generate_cauchy_samples(x, delta, k, N):
    n = len(x)
    samples = np.zeros((n, N))
    for i in range(n):
        samples[i] = cauchy.rvs(loc=x[i], scale=k * delta[i], size=N)
    return samples


# Оценка масштаба распределения Коши методом максимального правдоподобия
def estimate_scale_parameter(delta_y):
    def negative_log_likelihood(d):
        return -np.sum(np.log(cauchy.pdf(delta_y, scale=d)))

    result = minimize(negative_log_likelihood, x0=1, bounds=[(1e-6, None)])
    return result.x[0]


# Метод Крейновича + расчет доверительных интервалов для дисперсии с учетом наследственной погрешности
def kreynovich_method_variance(x, delta, k=0.01, N=1000, alpha=0.05):
    n = len(x)

    # Шаг 1: Вычислить исходные границы доверительного интервала для дисперсии
    lower_bound_0, upper_bound_0 = confidence_interval_variance(x, alpha)

    # Шаг 2: Сгенерировать новые значения x_ij из распределения Коши
    x_samples = generate_cauchy_samples(x, delta, k, N)

    # Шаг 3: Пересчитать доверительные интервалы для каждой комбинации (x_1j, ..., x_nj)
    lower_bounds = []
    upper_bounds = []
    for j in range(N):
        lb, ub = confidence_interval_variance(x_samples[:, j], alpha)
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Шаг 4: Оценить наследственную погрешность (отклонения от исходных границ)
    delta_lower = lower_bounds - lower_bound_0
    delta_upper = upper_bounds - upper_bound_0

    # Шаг 5: Оценить параметр масштаба d для нижней и верхней границ
    d_lower = estimate_scale_parameter(delta_lower)
    d_upper = estimate_scale_parameter(delta_upper)

    return {
        "Исходный доверительный интервал": (lower_bound_0, upper_bound_0),
        "Наследственная погрешность нижней границы": delta_lower,
        "Наследственная погрешность верхней границы": delta_upper,
        "Оценка масштаба для нижней границы": d_lower,
        "Оценка масштаба для верхней границы": d_upper
    }


# Пример использования
x = np.array([10, 12, 9, 11, 10.5])  # Исходные значения выборки
delta = np.full_like(x, 0.1)  # Неопределенности Delta_i

result = kreynovich_method_variance(x, delta, k=0.01, N=1000, alpha=0.05)

print("Результаты:")
print(f"Исходный доверительный интервал: {result['Исходный доверительный интервал']}")
print(f"Оценка масштаба для нижней границы: {result['Оценка масштаба для нижней границы']}")
print(f"Оценка масштаба для верхней границы: {result['Оценка масштаба для верхней границы']}")
