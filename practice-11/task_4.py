import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры задачи
alpha = 0.05  # Уровень значимости
beta = 0.05  # Уровень ошибки второго рода
sigma_squared = 3  # Константа для проверки гипотезы (дисперсия)
N = 10 ** 3  # Количество выборок
n_values = [10, 50, 100, 150, 200, 500]  # Размеры выборок
delta_sigma_values = np.arange(1.0, 6.2 + 0.4, 0.4)  # Диапазон сдвигов дисперсии

# Генерация выборок
np.random.seed(42)  # Фиксация случайности для воспроизводимости


def generate_samples(n, distribution, params, scale_adjust=1):
    if distribution == "normal":
        return np.random.normal(loc=params["mean"], scale=params["std"] * scale_adjust, size=(N, n))
    elif distribution == "uniform":
        return np.random.uniform(low=params["low"] * scale_adjust, high=params["high"] * scale_adjust, size=(N, n))
    elif distribution == "exponential":
        samples = np.random.exponential(scale=params["scale"] * scale_adjust, size=(N, n))
        return samples - params["shift"] * (scale_adjust - 1)


# Расчёт уровня значимости и доверительных интервалов
def one_sided_variance(samples, n, sigma_squared):
    sample_variances = np.var(samples, axis=1, ddof=1)  # Оценка дисперсии
    chi2_critical = stats.chi2.ppf(1 - alpha, df=n - 1)

    # Проверка гипотезы: дисперсия меньше sigma_squared
    reject_null = sample_variances * (n - 1) / sigma_squared < chi2_critical
    alpha_star = np.mean(reject_null)  # Достигнутый уровень значимости

    # Доверительные границы
    conf_interval = [
        alpha_star - stats.sem(reject_null),  # Нижняя граница
        alpha_star + stats.sem(reject_null)  # Верхняя граница
    ]
    return alpha_star, conf_interval


# Построение графиков для разных сдвигов дисперсии
def plot_alpha_vs_n(distribution, params):
    plt.figure(figsize=(12, 8))
    for delta_sigma in delta_sigma_values:
        alpha_stars = []
        lower_bounds = []
        upper_bounds = []

        for n in n_values:
            samples = generate_samples(n, distribution, params, scale_adjust=np.sqrt(delta_sigma))
            alpha_star, conf_interval = one_sided_variance(samples, n, sigma_squared)
            alpha_stars.append(alpha_star)
            lower_bounds.append(conf_interval[0])
            upper_bounds.append(conf_interval[1])

        # График уровня значимости
        plt.plot(n_values, alpha_stars, label=f"σ^2 shift = {delta_sigma:.1f}")

        # Доверительные интервалы
        plt.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2)

    plt.axhline(y=alpha, color="r", linestyle="--", label="α = 0.05")
    plt.title(f"Уровень значимости α* для {distribution.capitalize()} распределения")
    plt.xlabel("Размер выборки (n)")
    plt.ylabel("α*")
    plt.ylim(0, 0.5)  # Ограничение на ось y
    plt.legend()
    plt.grid()
    plt.show()


# Построение графика для пункта 2
def plot_constant_sigma(distribution, params):
    plt.figure(figsize=(12, 8))
    alpha_stars = []
    lower_bounds = []
    upper_bounds = []

    for n in n_values:
        samples = generate_samples(n, distribution, params)
        alpha_star, conf_interval = one_sided_variance(samples, n, sigma_squared)
        alpha_stars.append(alpha_star)
        lower_bounds.append(conf_interval[0])
        upper_bounds.append(conf_interval[1])

    # График уровня значимости
    plt.plot(n_values, alpha_stars, label="σ^2 = 3")

    # Доверительные интервалы
    plt.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2, label="Доверительный интервал")

    plt.axhline(y=alpha, color="r", linestyle="--", label="α = 0.05")
    plt.title(f"Уровень значимости α* для постоянной σ^2 ({distribution.capitalize()} распределение)")
    plt.xlabel("Размер выборки (n)")
    plt.ylabel("α*")
    plt.ylim(0, 1.0)  # Ограничение на ось y
    plt.legend()
    plt.grid()
    plt.show()


# Построение графиков для пункта 3 на одном листе (три распределения)
def plot_multiple_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    distributions = ["normal", "uniform", "exponential"]
    params_list = [params_normal, params_uniform, params_exponential]
    labels = ["Normal", "Uniform", "Exponential"]

    for ax, distribution, params, label in zip(axes, distributions, params_list, labels):
        for delta_sigma in delta_sigma_values:
            alpha_stars = []
            lower_bounds = []
            upper_bounds = []

            for n in n_values:
                samples = generate_samples(n, distribution, params, scale_adjust=np.sqrt(delta_sigma))
                alpha_star, conf_interval = one_sided_variance(samples, n, sigma_squared)
                alpha_stars.append(alpha_star)
                lower_bounds.append(conf_interval[0])
                upper_bounds.append(conf_interval[1])

            # График уровня значимости
            ax.plot(n_values, alpha_stars, label=f"σ^2 shift = {delta_sigma:.1f}")
            ax.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2)

        ax.axhline(y=alpha, color="r", linestyle="--")
        ax.set_title(f"{label} Distribution")
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("α*")
        ax.grid()

    # Общая легенда справа
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Legend")
    plt.subplots_adjust(right=0.85)
    plt.show()


# Параметры распределений
params_normal = {"mean": 0, "std": 1}
params_uniform = {"low": -np.sqrt(3), "high": np.sqrt(3)}
params_exponential = {"scale": 1, "shift": 1}

# Построение графика для пункта 2 (только одно распределение)
plot_constant_sigma("normal", params_normal)

# Построение графиков для пункта 3 (три распределения на одном листе)
plot_multiple_distributions()