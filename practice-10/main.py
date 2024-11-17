import numpy as np
from scipy.optimize import minimize

# Заданные параметры
N = 1000
x_values = np.arange(0.1, 3.1, 0.1)
a_true = np.array([0.0, 1.0, 1.0])


# Функция f(x, a)
def f(x, a0, a1, a2):
    return (a1 * x + a0) / (x + a2)


# Генерация данных y с разными типами шумов
def generate_data(x_values, noise_type, sigma=0.05):
    y_true = f(x_values, *a_true)
    if noise_type == "normal":
        noise = np.random.normal(0, sigma, len(x_values))
    elif noise_type == "uniform":
        noise = np.random.uniform(-sigma * np.sqrt(3), sigma * np.sqrt(3), len(x_values))
    elif noise_type == "laplace":
        lambda_param = 1 / (sigma * np.sqrt(2))
        noise = np.random.laplace(0, 1 / lambda_param, len(x_values))
    return y_true + noise


# Метод наименьших квадратов (OLS)
def ols_fit(x, y):
    def residuals(a):
        return np.sum((f(x, *a) - y) ** 2)

    result = minimize(residuals, a_true, method='Nelder-Mead')
    return result.x


# Метод LAD (Least Absolute Deviations)
def lad_fit(x, y):
    def residuals(a):
        return np.sum(np.abs(f(x, *a) - y))

    result = minimize(residuals, a_true, method='Nelder-Mead')
    return result.x


# Метод минимакс
def minimax_fit(x, y):
    def residuals(a):
        return np.max(np.abs(f(x, *a) - y))

    result = minimize(residuals, a_true, method='Nelder-Mead')
    return result.x


# Функция для удаления 5% крайних выбросов
def remove_outliers(data, percentile=5):
    lower_bound = np.percentile(data, percentile, axis=0)
    upper_bound = np.percentile(data, 100 - percentile, axis=0)
    filtered_data = np.array([row for row in data if np.all(row >= lower_bound) and np.all(row <= upper_bound)])
    return filtered_data


# Основной цикл для расчета оценок параметров a и дисперсий
results = {"ols": [], "lad": [], "minimax": []}
noise_types = ["normal", "uniform", "laplace"]

for noise_type in noise_types:
    print(f"\nШум: {noise_type}")
    estimates = {"ols": [], "lad": [], "minimax": []}

    for _ in range(N):
        y_data = generate_data(x_values, noise_type)

        # Получаем оценки параметров для каждого метода
        a_ols = ols_fit(x_values, y_data)
        a_lad = lad_fit(x_values, y_data)
        a_minimax = minimax_fit(x_values, y_data)

        # Сохраняем оценки
        estimates["ols"].append(a_ols)
        estimates["lad"].append(a_lad)
        estimates["minimax"].append(a_minimax)

    # Преобразуем оценки в массив
    for method in estimates:
        estimates_array = np.array(estimates[method])

        # Для распределения Лапласа удаляем 5% выбросов
        if noise_type == "laplace":
            estimates_array = remove_outliers(estimates_array)

        # Проверка, что данные имеют одинаковую длину после удаления выбросов
        if len(estimates_array) > 0:
            variance = np.var(estimates_array, axis=0)
            results[method].append(variance)
            print(f"Дисперсия оценок параметров для метода {method}: {variance}")
        else:
            print(
                f"Недостаточно данных для расчета дисперсии после удаления выбросов для метода {method} при шуме {noise_type}")

# Вывод результатов для анализа
for method in results:
    print(f"\nМетод {method}: Средняя дисперсия по всем типам шума - {np.mean(results[method], axis=0)}")
