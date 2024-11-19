import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Заданные параметры
N = 1000  # количество повторений
x_values = np.arange(0.1, 3.1, 0.1)  # значения x
a_true = np.array([0.0, 1.0, 1.0])  # истинные значения параметров
sigma = 0.05  # стандартное отклонение для нормального шума

# Уровни квантилей для выбросов
outlier_quantiles = [0.95, 0.99, 0.999]
outliers = [norm.ppf(q, loc=0, scale=sigma) for q in outlier_quantiles]


# Функция f(x, a)
def f(x, a0, a1, a2):
    return (a1 * x + a0) / (x + a2)


# Генерация данных y с нормальным шумом и добавлением выбросов
def generate_data_with_outliers(x_values, sigma=0.05, k=1):
    y_true = f(x_values, *a_true)
    noise = np.random.normal(0, sigma, len(x_values))
    y_data = y_true + noise

    # Добавляем k выбросов случайным образом
    for _ in range(k):
        index = np.random.randint(0, len(x_values))  # случайный индекс
        y_data[index] += np.random.choice(outliers)  # добавляем выброс

    return y_data


# Метод наименьших квадратов (OLS)
def ols_fit(x, y):
    def residuals(a):
        return np.sum((f(x, *a) - y) ** 2)

    result = minimize(residuals, a_true, method='Nelder-Mead')
    return result.x


# Метод наименьших абсолютных отклонений (LAR)
def lar_fit(x, y):
    def residuals(a):
        return np.sum(np.abs(f(x, *a) - y))

    result = minimize(residuals, a_true, method='Nelder-Mead')
    return result.x


# Основной цикл для оценки устойчивости регрессии с k выбросами
def calculate_outlier_robustness(k):
    ols_remaining_outliers = 0
    lar_remaining_outliers = 0

    for _ in range(N):
        y_data = generate_data_with_outliers(x_values, sigma, k=k)

        # Оценка параметров с помощью OLS и LAR
        a_ols = ols_fit(x_values, y_data)
        a_lar = lar_fit(x_values, y_data)

        # Проверяем, остались ли выбросы в данных
        if np.any(np.abs(f(x_values, *a_ols) - y_data) > np.max(outliers)):
            ols_remaining_outliers += 1
        if np.any(np.abs(f(x_values, *a_lar) - y_data) > np.max(outliers)):
            lar_remaining_outliers += 1

    ols_outlier_ratio = ols_remaining_outliers / N
    lar_outlier_ratio = lar_remaining_outliers / N
    return ols_outlier_ratio, lar_outlier_ratio


# Запуск оценки устойчивости для каждого количества выбросов
for k in range(1, 4):
    ols_ratio, lar_ratio = calculate_outlier_robustness(k)
    print(f"\nКоличество выбросов: {k}")
    print(f"Доля оставшихся выбросов (OLS): {ols_ratio:.2%}")
    print(f"Доля оставшихся выбросов (LAR): {lar_ratio:.2%}")
