import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры
alpha = 0.05
beta = 0.05
n_values = [10, 50, 100, 150, 200, 500]
delta_sigma_values = np.arange(1.0, 6.2 + 0.4, 0.4)
mu_values = np.arange(0, 3.2, 0.2)  # Варьируемое математическое ожидание
N = 1000  # Количество выборок
q = 0.99  # 99%-я квантиль
d = 3  # Пороговое значение


# Функция проверки гипотезы на основе квантиля
def one_sided_quantile(samples, n, d, q):
    sorted_samples = np.sort(samples, axis=1)  # Сортировка выборок
    k3 = np.clip(
        np.ceil(q * n + np.sqrt(q * (1 - q) * n) * stats.norm.ppf(1 - alpha)).astype(int) - 1,
        0,
        n - 1
    )
    k4 = np.clip(
        np.floor(q * n - np.sqrt(q * (1 - q) * n) * stats.norm.ppf(1 - beta)).astype(int) - 1,
        0,
        n - 1
    )

    reject_null = sorted_samples[:, k3] < d  # Гипотеза о квантиле
    alpha_star = np.mean(reject_null)  # Достигнутый уровень значимости

    # Доверительные границы
    conf_interval = [
        max(0, alpha_star - stats.sem(reject_null)),  # Нижняя граница
        min(1, alpha_star + stats.sem(reject_null))  # Верхняя граница
    ]
    return alpha_star, conf_interval


# Генерация данных
def generate_samples(n, distribution, params, mu_adjust=0):
    if distribution == "normal":
        return np.random.normal(loc=params["mean"] + mu_adjust, scale=params["std"], size=(N, n))
    elif distribution == "uniform":
        low = params["low"] + mu_adjust
        high = params["high"] + mu_adjust
        return np.random.uniform(low=low, high=high, size=(N, n))
    elif distribution == "exponential":
        return np.random.exponential(scale=params["scale"], size=(N, n)) + mu_adjust


# Построение графика для одного распределения
def plot_quantile_intervals(distribution, params):
    plt.figure(figsize=(12, 8))
    alpha_stars = []
    lower_bounds = []
    upper_bounds = []

    for n in n_values:
        samples = generate_samples(n, distribution, params)  # Генерация выборок
        alpha_star, conf_interval = one_sided_quantile(samples, n, d=d, q=q)  # Проверка гипотезы
        alpha_stars.append(alpha_star)
        lower_bounds.append(conf_interval[0])
        upper_bounds.append(conf_interval[1])

    # Построение линии с доверительными интервалами
    plt.plot(n_values, alpha_stars, label="α*")
    plt.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2, label="Доверительный интервал")

    # Линия для α = 0.05
    plt.axhline(y=alpha, color="r", linestyle="--", label="α = 0.05")

    plt.title(f"99%-я квантиль для {distribution.capitalize()} распределения")
    plt.xlabel("Размер выборки (n)")
    plt.ylabel("α*")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid()
    plt.show()


# Построение графиков для трех распределений
def plot_multiple_quantiles_varying_mu():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    distributions = ["normal", "uniform", "exponential"]
    params_list = [params_normal, params_uniform, params_exponential]
    labels = ["Normal", "Uniform", "Exponential"]

    for ax, distribution, params, label in zip(axes, distributions, params_list, labels):
        for mu in mu_values:
            alpha_stars = []
            lower_bounds = []
            upper_bounds = []

            for n in n_values:
                samples = generate_samples(n, distribution, params, mu_adjust=mu)
                alpha_star, conf_interval = one_sided_quantile(samples, n, d=d, q=q)
                alpha_stars.append(alpha_star)
                lower_bounds.append(conf_interval[0])
                upper_bounds.append(conf_interval[1])

            ax.plot(n_values, alpha_stars, label=f"μ = {mu:.1f}")
            ax.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2)

        ax.axhline(y=alpha, color="r", linestyle="--")
        ax.set_title(f"{label} Distribution")
        ax.set_xlabel("Размер выборки (n)")
        ax.set_ylabel("α*")
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Legend")
    plt.subplots_adjust(right=0.85)
    plt.show()


# Параметры распределений
params_normal = {"mean": 0, "std": 1}
params_uniform = {"low": -np.sqrt(3), "high": np.sqrt(3)}
params_exponential = {"scale": 1}

# Построение графика для одного распределения
plot_quantile_intervals("exponential", params_exponential)

# Построение графиков для трех распределений с варьированием математического ожидания
plot_multiple_quantiles_varying_mu()
