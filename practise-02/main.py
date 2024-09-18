import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Генераторы выборок
def generate_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N)


def generate_normal(N):
    return np.random.normal(0, 1, N)


def generate_sum_2_uniform_minus1_1(N):
    return np.random.uniform(-1, 1, N) + np.random.uniform(-1, 1, N)


# Функция для вычисления доверительных интервалов
def confidence_interval(sample, confidence=0.95):
    """Вычисляет доверительный интервал для выборки с заданным уровнем доверия."""
    mean = np.mean(sample)  # Вычисляет среднее значение выборки
    sem = stats.sem(sample)  # Вычисляет стандартную ошибку среднего
    interval = stats.t.interval(confidence, len(sample) - 1, loc=mean, scale=sem)  # Вычисляет доверительный интервал
    return interval


# Функция для генерации выборок и вычисления средних значений
def generate_samples_and_means(n_values):
    """Генерирует выборки и вычисляет средние значения для различных распределений."""
    means_uniform = []
    means_normal = []
    means_cauchy = []

    for n in n_values:  # Проходит по каждому значению n из списка n_values
        uniform_samples = generate_uniform_minus1_1(n)
        means_uniform.append(np.mean(uniform_samples))

        normal_samples = generate_normal(n)
        means_normal.append(np.mean(normal_samples))

        cauchy_samples = np.random.standard_cauchy(size=n)
        means_cauchy.append(np.mean(cauchy_samples))

    return means_uniform, means_normal, means_cauchy  # Возвращает списки средних значений


# Функция для визуализации средних значений
def plot_means(n_values, means_uniform, means_normal, means_cauchy):
    """Создает график средних значений для различных распределений."""
    plt.figure(figsize=(12, 6))  # Задает размер графика

    plt.subplot(1, 3, 1)
    plt.plot(n_values, means_uniform, marker='o', linestyle='-',
             color='b')
    plt.title('Среднее для равномерного распределения')
    plt.xlabel('n')
    plt.ylabel('Среднее значение')

    plt.subplot(1, 3, 2)  # Создает второй подграфик
    plt.plot(n_values, means_normal, marker='o', linestyle='-',
             color='g')
    plt.title('Среднее для нормального распределения')
    plt.xlabel('n')
    plt.ylabel('Среднее значение')

    plt.subplot(1, 3, 3)  # Создает третий подграфик
    plt.plot(n_values, means_cauchy, marker='o', linestyle='-',
             color='r')
    plt.title('Среднее для распределения Коши')
    plt.xlabel('n')
    plt.ylabel('Среднее значение')

    plt.tight_layout()
    plt.show()


n_values = [10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]  # Список различных размеров выборок в степени десяти
n_values_2 = [10, 100, 1000, 5000]  # Другой список размеров выборок для проверки доверительных интервалов
n_trials = 10000  # Количество испытаний для оценки доверительных интервалов
p = 0.95  # Уровень доверия для доверительных интервалов
true_mean = 0  # Истинное среднее значение (используется в проверке)

# Для хранения результатов
generators = {
    'uniform_minus1_1': generate_uniform_minus1_1,
    'normal': generate_normal,
    'sum_2_uniform_minus1_1': generate_sum_2_uniform_minus1_1
}  # Словарь генераторов выборок

interval_coverage = {gen: [] for gen in
                     generators}  # Словарь для хранения долей интервалов, содержащих истинное среднее

# Генерация выборок, расчет доверительных интервалов и проверка попадания 0 в интервалы
for gen_name, gen_func in generators.items():
    for N in n_values_2:  # Проходит по каждому размеру выборки из n_values_2
        count_included_zero = 0  # Счетчик попаданий истинного среднего в доверительные интервалы
        for _ in range(n_trials):  # Повторяет процесс n_trials раз
            sample = gen_func(N)  # Генерирует выборку размером N с помощью текущего генератора
            ci = confidence_interval(sample, confidence=p)  # Вычисляет доверительный интервал для выборки
            if ci[0] <= true_mean <= ci[1]:  # Проверяет, содержится ли истинное среднее в интервале
                count_included_zero += 1  # Увеличивает счетчик, если условие выполнено
        interval_coverage[gen_name].append(count_included_zero / n_trials)

# Вывод результатов
for gen_name in generators:  # Проходит по каждому генератору в словаре generators
    print(f"\nГенератор: {gen_name}")
    for i, N in enumerate(n_values_2):
        print(f"N={N}, Доля интервалов, содержащих 0: {interval_coverage[gen_name][i]:.4f}")
        # Выводит размер выборки и долю интервалов, содержащих истинное среднее

# Визуализация доверительных интервалов
plt.figure(figsize=(10, 6))
for gen_name in interval_coverage:
    plt.plot(n_values_2, interval_coverage[gen_name], label=gen_name)
    # Строит график долей интервалов для каждого генератора

plt.xscale('log')
plt.xlabel('Размер выборки (log scale)')
plt.ylabel('Доля интервалов, содержащих 0')
plt.title(f'Доверительные интервалы (p = {p}) для разных генераторов')

plt.legend()
plt.grid(True)
plt.show()

# Генерация выборок и вычисление средних значений
means_uniform, means_normal, means_cauchy = generate_samples_and_means(n_values)

# Визуализация средних значений
plot_means(n_values, means_uniform, means_normal, means_cauchy)
