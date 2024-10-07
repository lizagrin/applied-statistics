import numpy as np
import scipy.stats as stats


# Функции для генерации выборок разных распределений
def generate_sample(distribution, n):
    if distribution == 'uniform':
        return np.random.uniform(-1, 1, n)
    elif distribution == 'normal':
        return np.random.normal(0, 1, n)
    elif distribution == 'sum_of_uniforms':
        return np.random.uniform(-1, 1, n) + np.random.uniform(-1, 1, n)


# Функция для оценки достаточного размера выборки
def sufficient_sample_size(sample, delta, Q=0.95):
    n_min = 5
    # по формуле нужно найти распределение Пирсона, дисперсию, коэффициент Стьюдента
    chi2_upper = stats.chi2.ppf((1 + Q) / 2, df=n_min - 1)
    chi2_lower = stats.chi2.ppf((1 - Q) / 2, df=n_min - 1)

    D = np.var(sample, ddof=1)

    sigma_1 = np.sqrt(D * (n_min - 1) / chi2_upper)
    sigma_2 = np.sqrt(D * (n_min - 1) / chi2_lower)

    t_student = stats.t.ppf((1 + Q) / 2, df=n_min - 1)

    # Рассчет n для sigma_1
    n_1 = (sigma_1 ** 2 * (t_student ** 2)) / (delta ** 2)

    # Рассчет n для sigma_2 (оценка сверху)
    n_2 = (sigma_2 ** 2 * (t_student ** 2)) / (delta ** 2)

    return int(np.ceil(n_1)), int(np.ceil(n_2))


# Основной код для разных значений Delta
def experiment(distribution, deltas):
    results = {}

    for delta in deltas:
        sample = generate_sample(distribution, 5)
        n_iterative, n_upper = sufficient_sample_size(sample, delta)

        # Итерационная процедура: уточняем размер выборки несколько раз
        for i in range(3):  # 3 итерации
            sample = generate_sample(distribution, n_iterative)
            n_iterative, _ = sufficient_sample_size(sample, delta)

        results[delta] = {'n_iterative': n_iterative, 'n_upper': n_upper}

    return results


# Функция для форматированного вывода результатов
def print_results(distribution_name, results):
    print(f"Распределение: {distribution_name}")
    for delta, result in results.items():
        print(f"Delta: {delta}, итеративный метод: {result['n_iterative']}, оценка сверху: {result['n_upper']} ")
    print()


# Запуск для трех распределений
deltas = [10 ** -1, 10 ** -2, 10 ** -3]

uniform_results = experiment('uniform', deltas)
normal_results = experiment('normal', deltas)
sum_uniform_results = experiment('sum_of_uniforms', deltas)

# вывод
print_results("Uniform(-1, 1)", uniform_results)
print_results("Normal(0, 1)", normal_results)
print_results("Sum of Two Uniform(-1, 1)", sum_uniform_results)
