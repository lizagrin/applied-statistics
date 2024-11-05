class Dual:
    def __init__(self, real, dual=1.0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        else:
            return Dual(self.real + other, self.dual)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return Dual(self.real * other, self.dual * other)

    def __pow__(self, power):
        return Dual(self.real ** power, power * self.real ** (power - 1) * self.dual)

    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual})"


def function_f1(inputs):
    # f(x1, x2, ..., x10) = sum(x_i ** 2)
    result = Dual(0, 0)
    for x in inputs:
        result += x ** 2
    return result


def function_f2(inputs):
    # f(x1, x2, ..., x10) = sum(x_i ** x_i)
    result = Dual(0, 0)
    for x in inputs:
        result += x ** x.real  # Используем Dual степень real компоненты самого себя
    return result


def compute_gradients(func, x_vals):
    gradients = []
    for i in range(len(x_vals)):
        dual_inputs = [Dual(x, 1 if i == j else 0) for j, x in enumerate(x_vals)]
        result = func(dual_inputs)
        gradients.append(result.dual)
    return gradients
