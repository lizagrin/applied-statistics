import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры
N = 1000
n_values = [10, 50, 100, 150, 200, 500]
alpha = 0.05

# Генерация данных
np.random.seed(42)


def generate_samples(distribution, n, N):
    if distribution == 'normal':
        return np.random.normal(loc=0, scale=1, size=(N, n))
    elif distribution == 'uniform':
        return np.random.uniform(low=-1, high=1, size=(N, n))
    elif distribution == 'exponential':
        return np.random.exponential(scale=1, size=(N, n))
    elif distribution == 'beta':
        return np.random.beta(a=2, b=5, size=(N, n))
    elif distribution == 'cauchy':
        return np.random.standard_cauchy(size=(N, n))
    elif distribution == 'lognormal':
        return np.random.lognormal(mean=0, sigma=1, size=(N, n))


# Проверка гипотезы и вычисление alpha*
def perform_tests(samples, alpha):
    sw_rejections = []
    sf_rejections = []

    for sample in samples:
        # Шапиро-Уилк
        sw_stat, sw_p = stats.shapiro(sample)
        sw_rejections.append(sw_p < alpha)

        # Шапиро-Франция через критерий Андерсона-Дарлинга
        sf_stat, critical_values, significance_levels = stats.anderson(sample, dist='norm')
        sf_rejections.append(sf_stat > critical_values[np.where(significance_levels == alpha * 100)[0][0]])

    sw_rejection_rate = np.mean(sw_rejections)
    sf_rejection_rate = np.mean(sf_rejections)

    return sw_rejection_rate, sf_rejection_rate


# График уровня значимости
plt.figure(figsize=(12, 8))
colors = {
    'normal': 'blue',
    'uniform': 'green',
    'exponential': 'orange',
    'beta': 'purple',
    'cauchy': 'brown',
    'lognormal': 'pink'
}

for distribution in ['normal', 'uniform', 'exponential', 'beta', 'cauchy', 'lognormal']:
    sw_results = []
    sf_results = []
    sw_conf_intervals = []
    sf_conf_intervals = []

    for n in n_values:
        samples = generate_samples(distribution, n, N)
        sw_rate, sf_rate = perform_tests(samples, alpha)

        # Вычисление доверительных интервалов
        sw_se = np.sqrt((sw_rate * (1 - sw_rate)) / N)
        sf_se = np.sqrt((sf_rate * (1 - sf_rate)) / N)

        sw_ci_lower = sw_rate - 1.96 * sw_se
        sw_ci_upper = sw_rate + 1.96 * sw_se

        sf_ci_lower = sf_rate - 1.96 * sf_se
        sf_ci_upper = sf_rate + 1.96 * sf_se

        sw_results.append(sw_rate)
        sf_results.append(sf_rate)
        sw_conf_intervals.append((sw_ci_lower, sw_ci_upper))
        sf_conf_intervals.append((sf_ci_lower, sf_ci_upper))

    # Графики для SW и SF
    sw_results = np.array(sw_results)
    sf_results = np.array(sf_results)
    sw_conf_intervals = np.array(sw_conf_intervals)
    sf_conf_intervals = np.array(sf_conf_intervals)

    plt.plot(n_values, sw_results, label=f"SW {distribution}", linestyle='-', color=colors[distribution])
    plt.fill_between(n_values, sw_conf_intervals[:, 0], sw_conf_intervals[:, 1], alpha=0.2, color=colors[distribution])

    plt.plot(n_values, sf_results, label=f"SF {distribution}", linestyle='--', color=colors[distribution])
    plt.fill_between(n_values, sf_conf_intervals[:, 0], sf_conf_intervals[:, 1], alpha=0.2, color=colors[distribution])

plt.axhline(y=alpha, color='r', linestyle='--', label=f"alpha = {alpha}")
plt.xlabel("Размер выборки (n)")
plt.ylabel("alpha*")
plt.title("Уровень значимости для различных распределений")
plt.legend()
plt.grid()
plt.show()
