import numpy as np
import matplotlib.pyplot as plt


def generate_samples(N):
    # Равномерное распределение от 0 до 1
    uniform_0_1 = np.random.uniform(0, 1, N)

    # Равномерное распределение от -1 до 1
    uniform_neg1_1 = np.random.uniform(-1, 1, N)

    # Распределение суммы трех случайных величин, распределенных равномерно на [-1, 1]
    sum_of_three_uniforms = np.random.uniform(-1, 1, (3, N)).sum(axis=0)

    # Распределение Коши с параметром сдвига 0 и масштаба 1
    cauchy_distribution = np.random.standard_cauchy(N)

    return uniform_0_1, uniform_neg1_1, sum_of_three_uniforms, cauchy_distribution


def plot_histograms(N):
    # Генерация выборок
    uniform_0_1, uniform_neg1_1, sum_of_three_uniforms, cauchy_distribution = generate_samples(N)

    # Создание графиков
    plt.figure(figsize=(20, 5))

    # Гистограмма для равномерного распределения от 0 до 1
    plt.subplot(1, 4, 1)
    plt.hist(uniform_0_1, bins=50, color='lightblue', edgecolor='black')
    plt.title('Равномерное распределение от 0 до 1')
    plt.xlabel('Значение')
    plt.ylabel('Частота')

    # Гистограмма для равномерного распределения от -1 до 1
    plt.subplot(1, 4, 2)
    plt.hist(uniform_neg1_1, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Равномерное распределение от -1 до 1')
    plt.xlabel('Значение')
    plt.ylabel('Частота')

    # Гистограмма для распределения суммы трех случайных величин
    plt.subplot(1, 4, 3)
    plt.hist(sum_of_three_uniforms, bins=50, color='salmon', edgecolor='black')
    plt.title('Распределение суммы трех СВ от -1 до 1')
    plt.xlabel('Значение')
    plt.ylabel('Частота')

    # Гистограмма для распределения Коши
    plt.subplot(1, 4, 4)
    plt.hist(cauchy_distribution, bins=50, color='violet', edgecolor='black', range=(-25, 25))
    plt.title('Распределение Коши ')
    plt.xlabel('Значение')
    plt.ylabel('Частота')

    # Отображение графиков
    plt.tight_layout()
    plt.show()


# Пример использования
N = 1000
plot_histograms(N)
