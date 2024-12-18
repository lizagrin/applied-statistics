import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, newton

# Заданные параметры для уравнений и их погрешности
a2_f1_range = [1.0, 1.1]
a1_f1_range = [-0.1, 0.1]
a0_f1_range = [-0.1, 0.1]

a2_f2_range = [1.0, 1.1]
a1_f2_range = [1.9, 2.1]
a0_f2_range = [1.1, 1.2]


# Уравнение 1
def f1(x, a0, a1, a2):
    return a2 * x ** 2 + a1 * x + a0


# Уравнение 2
def f2(x, a0, a1, a2):
    return np.exp(a2 * x) + a1 * x + a0


# Проверка существования корней на интервале
def has_root(func, a, b, args):
    return func(a, *args) * func(b, *args) <= 0


# Проверка дискриминанта для f1
def check_discriminant(a0, a1, a2):
    return a1 ** 2 - 4 * a2 * a0 >= 0


# 1. Метод Монте-Карло с проверкой на существование корней
def monte_carlo_solver(func, a_ranges, N=1000, tol=1e-5):
    roots = []
    for _ in range(N):
        a0 = np.random.uniform(*a_ranges["a0"])
        a1 = np.random.uniform(*a_ranges["a1"])
        a2 = np.random.uniform(*a_ranges["a2"])
        args = (a0, a1, a2)
        if func == f1 and not check_discriminant(a0, a1, a2):
            continue
        if has_root(func, -10, 10, args):
            try:
                root = bisect(func, -10, 10, args=args, xtol=tol)
                roots.append(root)
            except ValueError:
                pass
    return roots


# 2. Интервальная бисекция с построением графика интервалов
def interval_bisection(func, a_ranges, tol=1e-5):
    a0_mid = np.mean(a_ranges["a0"])
    a1_mid = np.mean(a_ranges["a1"])
    a2_mid = np.mean(a_ranges["a2"])
    args = (a0_mid, a1_mid, a2_mid)
    intervals = [(-10, 10)]
    history = []

    if func == f1 and not check_discriminant(a0_mid, a1_mid, a2_mid):
        return [], history
    if not has_root(func, -10, 10, args):
        return [], history

    for _ in range(20):
        new_intervals = []
        for interval in intervals:
            mid = (interval[0] + interval[1]) / 2
            if has_root(func, interval[0], mid, args):
                new_intervals.append((interval[0], mid))
            if has_root(func, mid, interval[1], args):
                new_intervals.append((mid, interval[1]))
        intervals = new_intervals
        history.append(intervals)
        if all(abs(b - a) < tol for a, b in intervals):
            break
    return intervals, history


# 3. Аналитическое решение для f1
def analytical_solution_f1(a_ranges):
    a0 = np.mean(a_ranges["a0"])
    a1 = np.mean(a_ranges["a1"])
    a2 = np.mean(a_ranges["a2"])

    discriminant = a1 ** 2 - 4 * a2 * a0
    if discriminant < 0:
        print("Дискриминант отрицательный, корней нет.")
        return None, None

    root1 = (-a1 + np.sqrt(discriminant)) / (2 * a2)
    root2 = (-a1 - np.sqrt(discriminant)) / (2 * a2)
    return root1, root2


# 4. Аналитическое решение для f2
def analytical_solution_f2(a_ranges, tol=1e-5):
    a0 = np.mean(a_ranges["a0"])
    a1 = np.mean(a_ranges["a1"])
    a2 = np.mean(a_ranges["a2"])

    def func_f2(x):
        return np.exp(a2 * x) + a1 * x + a0

    x0 = -1.0  # Начальное приближение
    try:
        root = newton(func_f2, x0, tol=tol)
    except RuntimeError:
        print("Не удалось найти корень для f2.")
        return None
    return root


# Запуск методов для обеих функций
def solve_and_analyze(func, params, func_name):
    print(f"\nРезультаты для {func_name}:")

    # Метод Монте-Карло
    roots_mc = monte_carlo_solver(func, params)
    if roots_mc:
        mean_root = np.mean(roots_mc)
        print(f"Средний корень методом Монте-Карло: {mean_root:.5f}")
    else:
        print("Корни не найдены методом Монте-Карло.")

    # Интервальная бисекция
    intervals, history = interval_bisection(func, params)
    if intervals:
        print(f"Интервалы для корня после бисекции: {intervals}")

        # Построение графика изменения интервалов
        plt.figure(figsize=(10, 6))
        for i, step in enumerate(history):
            for interval in step:
                plt.plot([i, i], interval, 'bo-', markersize=5)
        plt.title(f"Изменение интервалов для {func_name} методом бисекции")
        plt.xlabel("Итерация")
        plt.ylabel("Интервал")
        plt.grid()
        plt.show()
    else:
        print("Корни не найдены методом бисекции.")

    # Аналитическое решение
    if func == f1:
        roots = analytical_solution_f1(params)
        if roots:
            print(f"Аналитическое решение: x1 = {roots[0]:.5f}, x2 = {roots[1]:.5f}")
    elif func == f2:
        root = analytical_solution_f2(params)
        if root is not None:
            print(f"Аналитическое решение для f2: x = {root:.5f}")


# Параметры функций
params_f1 = {"a0": a0_f1_range, "a1": a1_f1_range, "a2": a2_f1_range}
params_f2 = {"a0": a0_f2_range, "a1": a1_f2_range, "a2": a2_f2_range}

# Решение и анализ для f1 и f2
solve_and_analyze(f1, params_f1, "f1")
solve_and_analyze(f2, params_f2, "f2")
