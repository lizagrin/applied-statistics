import time
from itertools import product

import numpy as np

from backward_autodiff import grad_f1, grad_f2

# Определение значений для функции f1
x_values_f1 = np.linspace(0.0, 1.0, 11)

# Перебор всех комбинаций для функции f1
start_time = time.time()
for x_vals in product(x_values_f1, repeat=6):
    grad_f1(np.array(x_vals))
print("Время выполнения для функции f1 с backward mode:", time.time() - start_time, "секунд")

# Определение значений для функции f2
x_values_f2 = np.linspace(1.0, 2.0, 11)

# Перебор всех комбинаций для функции f2
start_time = time.time()
for x_vals in product(x_values_f2, repeat=6):
    grad_f2(np.array(x_vals))
print("Время выполнения для функции f2 с backward mode:", time.time() - start_time, "секунд")
