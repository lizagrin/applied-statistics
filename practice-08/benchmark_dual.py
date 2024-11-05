import time
from itertools import product

import numpy as np

from dual_autodiff import function_f1, function_f2, compute_gradients

# Набор значений для переменных x1, x2, ..., x6
x_values = np.arange(0.0, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0] для функции f1
x_values_f2 = np.arange(1.0, 2.1, 0.1)  # [1.0, 1.1, 1.2, ..., 2.0] для функции f2

# Подсчет времени для полной переборки всех сочетаний для функции f1
start_time = time.time()
for x_vals in product(x_values, repeat=6):
    compute_gradients(function_f1, list(x_vals))
print("Время выполнения для функции f1 с dual numbers:", time.time() - start_time, "секунд")

# Подсчет времени для полной переборки всех сочетаний для функции f2
start_time = time.time()
for x_vals in product(x_values_f2, repeat=6):
    compute_gradients(function_f2, list(x_vals))
print("Время выполнения для функции f2 с dual numbers:", time.time() - start_time, "секунд")
