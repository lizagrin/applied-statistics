import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

# Система уравнений
# a1 * x + a2 * y = b1
# a3 * x^2 + a4 * y^2 = b2
def system_of_equations(vars, a1, a2, a3, a4, b1, b2):
    x, y = vars
    eq1 = a1 * x + a2 * y - b1
    eq2 = a3 * x**2 + a4 * y**2 - b2
    return [eq1, eq2]

# Метод Ньютона с использованием встроенной библиотеки
def newton_solver(a_ranges, b_ranges, tol=1e-5, max_iter=100):
    a1 = np.mean(a_ranges["a1"])
    a2 = np.mean(a_ranges["a2"])
    a3 = np.mean(a_ranges["a3"])
    a4 = np.mean(a_ranges["a4"])
    b1 = np.mean(b_ranges["b1"])
    b2 = np.mean(b_ranges["b2"])

    # Начальное приближение
    x0, y0 = 0.5, 0.5

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = fsolve(system_of_equations, [x0, y0], args=(a1, a2, a3, a4, b1, b2), xtol=tol)
        return sol
    except Exception as e:
        print(f"Ошибка решения методом Ньютона: {e}")
        return None

# Метод Монте-Карло для системы уравнений
def monte_carlo_solver_system(a_ranges, b_ranges, N=1000):
    solutions = []
    for _ in range(N):
        a1 = np.random.uniform(*a_ranges["a1"])
        a2 = np.random.uniform(*a_ranges["a2"])
        a3 = np.random.uniform(*a_ranges["a3"])
        a4 = np.random.uniform(*a_ranges["a4"])
        b1 = np.random.uniform(*b_ranges["b1"])
        b2 = np.random.uniform(*b_ranges["b2"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = fsolve(system_of_equations, [1, 1], args=(a1, a2, a3, a4, b1, b2))
            solutions.append(sol)
        except:
            pass
    return np.array(solutions)

# Аналитическая оценка погрешности для системы уравнений
# Решение аналитическим методом и вычисление погрешности
def analytical_error_system(a_ranges, b_ranges):
    a1 = np.mean(a_ranges["a1"])
    a2 = np.mean(a_ranges["a2"])
    a3 = np.mean(a_ranges["a3"])
    a4 = np.mean(a_ranges["a4"])
    b1 = np.mean(b_ranges["b1"])
    b2 = np.mean(b_ranges["b2"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root_guess = fsolve(system_of_equations, [0.5, 0.5], args=(a1, a2, a3, a4, b1, b2))

    delta_a1 = (a_ranges["a1"][1] - a_ranges["a1"][0]) / 2
    delta_a2 = (a_ranges["a2"][1] - a_ranges["a2"][0]) / 2
    delta_a3 = (a_ranges["a3"][1] - a_ranges["a3"][0]) / 2
    delta_a4 = (a_ranges["a4"][1] - a_ranges["a4"][0]) / 2
    delta_b1 = (b_ranges["b1"][1] - b_ranges["b1"][0]) / 2
    delta_b2 = (b_ranges["b2"][1] - b_ranges["b2"][0]) / 2

    x, y = root_guess
    error_x = abs(delta_a1 * x + delta_a2 * y + delta_b1)
    error_y = abs(delta_a3 * x**2 + delta_a4 * y**2 + delta_b2)

    return root_guess, (error_x, error_y)

# Параметры с неопределённостью
a_ranges = {"a1": [1.0, 1.1], "a2": [1.0, 1.2], "a3": [0.9, 1.0], "a4": [0.8, 0.9]}
b_ranges = {"b1": [2.0, 2.1], "b2": [1.5, 1.6]}

# Решение методом Ньютона
newton_solution = newton_solver(a_ranges, b_ranges)
if newton_solution is not None:
    print(f"Решение методом Ньютона: x = {newton_solution[0]:.5f}, y = {newton_solution[1]:.5f}")
else:
    print("Метод Ньютона не нашел решение.")

# Решение методом Монте-Карло
solutions_mc = monte_carlo_solver_system(a_ranges, b_ranges)
if solutions_mc.size > 0:
    mean_solution = np.mean(solutions_mc, axis=0)
    print(f"Среднее решение методом Монте-Карло: x = {mean_solution[0]:.5f}, y = {mean_solution[1]:.5f}")

# Решение аналитическим методом
root_analytical, (error_x, error_y) = analytical_error_system(a_ranges, b_ranges)
print(f"Решение аналитическим методом: x = {root_analytical[0]:.5f}, y = {root_analytical[1]:.5f}")
print(f"Аналитическая погрешность: x = {error_x:.5f}, y = {error_y:.5f}")

# Визуализация метода Монте-Карло
if solutions_mc.size > 0:
    plt.figure(figsize=(8, 6))
    plt.scatter(solutions_mc[:, 0], solutions_mc[:, 1], alpha=0.5, label="Метод Монте-Карло", color='b')
    plt.axvline(mean_solution[0], color='r', linestyle='--', label=f"Среднее x = {mean_solution[0]:.5f}")
    plt.axhline(mean_solution[1], color='r', linestyle='--', label=f"Среднее y = {mean_solution[1]:.5f}")
    plt.title("Распределение решений методом Монте-Карло")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
