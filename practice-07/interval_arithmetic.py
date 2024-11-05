import numpy as np


# Класс для классической интервальной арифметики
class Interval:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        else:
            return Interval(self.lower + other, self.upper + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        else:
            return Interval(self.lower - other, self.upper - other)

    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper,
            ]
            return Interval(min(products), max(products))
        else:
            return Interval(self.lower * other, self.upper * other)

    def __truediv__(self, other):
        if isinstance(other, Interval):
            if other.lower <= 0 <= other.upper:
                raise ZeroDivisionError("Interval includes zero, division undefined.")
            reciprocals = [
                1 / other.lower,
                1 / other.upper
            ]
            return self * Interval(min(reciprocals), max(reciprocals))
        else:
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return Interval(self.lower / other, self.upper / other)

    def __repr__(self):
        return f"[{self.lower}, {self.upper}]"


# Класс для аффинной арифметики
class AffineInterval:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __add__(self, other):
        if isinstance(other, AffineInterval):
            return AffineInterval(self.center + other.center, self.radius + other.radius)
        else:
            return AffineInterval(self.center + other, self.radius)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, AffineInterval):
            return AffineInterval(self.center - other.center, self.radius + other.radius)
        else:
            return AffineInterval(self.center - other, self.radius)

    def __mul__(self, other):
        if isinstance(other, AffineInterval):
            new_center = self.center * other.center
            new_radius = abs(self.center * other.radius) + abs(self.radius * other.center) + self.radius * other.radius
            return AffineInterval(new_center, new_radius)
        else:
            new_center = self.center * other
            new_radius = abs(self.radius * other)
            return AffineInterval(new_center, new_radius)

    def __truediv__(self, other):
        if isinstance(other, AffineInterval):
            if other.center == 0:
                raise ZeroDivisionError("Division by zero.")
            new_center = self.center / other.center
            new_radius = (abs(self.radius / other.center) +
                          abs(self.center * other.radius / (other.center ** 2)))
            return AffineInterval(new_center, new_radius)
        else:
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return AffineInterval(self.center / other, self.radius / abs(other))

    def __repr__(self):
        return f"{self.center} ± {self.radius}"


# Интервальная версия функции f
def f_interval(x):
    norm_squared = sum([xi * xi for xi in x])  # Норма в квадрате
    exp_part = Interval(np.exp(-norm_squared.upper), np.exp(-norm_squared.lower))
    return [exp_part * xi for xi in x]


# Аффинная версия функции f
def f_affine(x):
    norm_squared = sum([xi * xi for xi in x])  # Норма в квадрате
    exp_part = AffineInterval(np.exp(-norm_squared.center - norm_squared.radius),
                              np.exp(-norm_squared.center + norm_squared.radius))
    return [exp_part * xi for xi in x]


# Выполнение расчетов с классической интервальной арифметикой
x_interval = [Interval(1, 1.1), Interval(1, 1.1), Interval(1, 1.1)]
result_interval = f_interval(x_interval)
print("Результат с классической интервальной арифметикой:", result_interval)

# Выполнение расчетов с аффинной арифметикой
x_affine = [AffineInterval(1, 0.1), AffineInterval(1, 0.1), AffineInterval(1, 0.1)]
result_affine = f_affine(x_affine)
print("Результат с аффинной арифметикой:", result_affine)
