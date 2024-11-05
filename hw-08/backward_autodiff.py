import autograd.numpy as np
from autograd import grad


def function_f1(inputs):
    return np.sum(inputs ** 2)


def function_f2(inputs):
    return np.sum(inputs ** inputs)


# Вычисление градиента функции f1
grad_f1 = grad(function_f1)

# Вычисление градиента функции f2
grad_f2 = grad(function_f2)
