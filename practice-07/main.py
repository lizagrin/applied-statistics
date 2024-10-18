import numpy as np
import matplotlib.pyplot as plt


# Определение функции f
def f(x):
    # Функция f(x) = exp(-x^T * x) * x, где x — это вектор.
    # np.dot(x, x) — скалярное произведение вектора x на себя (равно ||x||^2).
    # np.exp(-np.dot(x, x)) — экспонента от минус нормы вектора в квадрате.
    # Произведение этого скалярного значения на сам вектор x.
    return np.exp(-np.dot(x, x)) * x


# Метод конечной разности
def finite_difference(f, x, h):
    # Создаем массив для градиента такой же размерности, как и x, и заполняем единицами.
    grad = np.ones_like(x)
    # Цикл по каждому элементу вектора x
    for i in range(len(x)):
        # Создаем копию вектора x для изменения конкретного элемента
        x_forward = np.copy(x)
        # Увеличиваем i-й элемент на h
        x_forward[i] += h
        # Вычисляем частную производную по i-му элементу
        # Используем конечную разность: (f(x+h) - f(x)) / h
        grad[i] = (f(x_forward) - f(x))[i] / h
    # Возвращаем вектор градиентов
    return grad


# Метод комплексного приращения
def complex_step(f, x, h):
    # Создаем массив для градиента такой же размерности, как и x, и заполняем нулями.
    grad = np.zeros_like(x)
    # Цикл по каждому элементу вектора x
    for i in range(len(x)):
        # Создаем копию вектора x и приводим его к комплексному типу
        x_complex = np.copy(x).astype(np.complex128)
        # Добавляем к i-му элементу мнимую часть h
        x_complex[i] += 1j * h
        # Вычисляем частную производную по i-му элементу
        # Используем метод комплексного приращения: Im(f(x + ih)) / h
        grad[i] = np.imag(f(x_complex))[i] / h
    # Возвращаем вектор градиентов
    return grad


# Списки для сохранения значений производных для каждого метода
finite_diffs = []
complex_steps = []
# Начальные условия
x = np.zeros(3)  # Вектор нулевых значений (можно выбрать любое количество элементов)
h_values = [10 ** (-p) for p in range(300, -1, -1)]  # Массив значений alpha = 10^(-p)

# Вычисления
for h in h_values:
    finite_diff_grad = finite_difference(f, x, h)
    complex_step_grad = complex_step(f, x, h)
    finite_diffs.append(finite_diff_grad[0])  # Сохраняем значение по первому аргументу
    complex_steps.append(complex_step_grad[0])
    print(f"h = {h}")
    print(f"Конечная разность: {finite_diff_grad}")
    print(f"Комплексное приращение: {complex_step_grad}")

# Преобразуем в массивы для удобства графиков
finite_diffs = np.array(finite_diffs)
complex_steps = np.array(complex_steps)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(np.log10(h_values), finite_diffs, label="Конечная разность", color='r')
plt.plot(np.log10(h_values), complex_steps, label="Комплексное приращение", color='b')
plt.xlabel("log10(h)")
plt.ylabel("Оценка производной")
plt.title("Зависимость оценки производной от значения h")
plt.legend()
plt.grid(True)
plt.show()
