import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры задачи
alpha = 0.05  # Уровень значимости
beta = 0.05  # Уровень ошибки второго рода
t = 5  # Параметр интервала
N = 10 ** 3  # Количество выборок
n_values = [10, 50, 100, 150, 200, 500]  # Размеры выборок
sigma_values = np.arange(1.0, t + 0.2, 0.2)  # Шаг изменения σ

np.random.seed(42)  # Для воспроизводимости


# Генерация выборок
def generate_samples(n, distribution, params, scale_adjust=1):
    if distribution == "normal":
        return np.random.normal(loc=params["mean"], scale=params["std"] * scale_adjust, size=(N, n))
    elif distribution == "uniform":
        return np.random.uniform(low=params["low"] * scale_adjust, high=params["high"] * scale_adjust, size=(N, n))
    elif distribution == "exponential":
        samples = np.random.exponential(scale=params["scale"] * scale_adjust, size=(N, n))
        return samples - params["shift"] * (scale_adjust - 1)


# Проверка гипотезы по интерквантильному промежутку
def check_interquantile(samples, t):
    lower_q, upper_q = np.percentile(samples, [2.5, 97.5], axis=1)  # 95%-интерквантили
    within_bounds = (lower_q >= -t) & (upper_q <= t)  # Условие укладывания в интервал
    alpha_star = np.mean(within_bounds)  # Достигнутый уровень значимости

    # Доверительный интервал для alpha*
    conf_interval = [
        alpha_star - stats.sem(within_bounds),
        alpha_star + stats.sem(within_bounds)
    ]
    return alpha_star, conf_interval


def plot_fixed_t(distribution, params):
    plt.figure(figsize=(12, 8))
    alpha_stars = []
    lower_bounds = []
    upper_bounds = []

    for n in n_values:
        samples = generate_samples(n, distribution, params)
        alpha_star, conf_interval = check_interquantile(samples, t=5)  # t фиксировано
        alpha_stars.append(alpha_star)
        lower_bounds.append(conf_interval[0])
        upper_bounds.append(conf_interval[1])

    # График уровня значимости
    plt.plot(n_values, alpha_stars, label=f"t = 5")

    # Доверительные интервалы
    plt.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2, label="Доверительный интервал")

    plt.axhline(y=alpha, color="r", linestyle="--", label="α = 0.05")
    plt.title(f"Уровень значимости α* для {distribution.capitalize()} распределения (t = 5)")
    plt.xlabel("Размер выборки (n)")
    plt.ylabel("α*")
    plt.ylim(0, 1.0)  # Ограничение на ось y
    plt.legend()
    plt.grid()
    plt.show()

def plot_multiple_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    distributions = ["normal", "uniform", "exponential"]
    params_list = [params_normal, params_uniform, params_exponential]
    labels = ["Normal", "Uniform", "Exponential"]

    for ax, distribution, params, label in zip(axes, distributions, params_list, labels):
        for sigma in sigma_values:
            alpha_stars = []
            lower_bounds = []
            upper_bounds = []

            for n in n_values:
                samples = generate_samples(n, distribution, params, scale_adjust=sigma)
                alpha_star, conf_interval = check_interquantile(samples, t)
                alpha_stars.append(alpha_star)
                lower_bounds.append(conf_interval[0])
                upper_bounds.append(conf_interval[1])

            # График уровня значимости
            ax.plot(n_values, alpha_stars, label=f"σ = {sigma:.1f}")
            ax.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2)

        ax.axhline(y=alpha, color="r", linestyle="--")
        ax.set_title(f"{label} Distribution")
        ax.set_xlabel("Размер выборки (n)")
        ax.set_ylabel("α*")
        ax.grid()

    # Общая легенда справа
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Легенда")
    plt.subplots_adjust(right=0.85)
    plt.show()


# Параметры распределений
params_normal = {"mean": 0, "std": 1}
params_uniform = {"low": -np.sqrt(3), "high": np.sqrt(3)}
params_exponential = {"scale": 1, "shift": 1}

# Построение графиков
plot_fixed_t("normal", params_normal)  # Для нормального распределения
plot_multiple_distributions()  # Для всех трёх распределений
