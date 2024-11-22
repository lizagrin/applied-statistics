import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры задачи
alpha = 0.05  # Уровень значимости
a = 5  # Константа для проверки гипотезы
N = 10 ** 3  # Количество выборок
n_values = [10, 50, 100, 150, 200, 500]  # Размеры выборок
delta_mu_values = np.arange(0.2, 2 * a + 0.2, 0.2)  # Сдвиги математического ожидания


# Параметры распределений
params_normal = {"mean": 0, "std": 1}
params_uniform = {"low": -np.sqrt(3), "high": np.sqrt(3)}
params_exponential = {"scale": 1, "shift": 1}


# Функция для генерации выборок с учетом сдвига математического ожидания
def generate_samples_shifted(n, distribution, params, delta_mu):
    if distribution == "normal":
        return np.random.normal(loc=params["mean"] + delta_mu, scale=params["std"], size=(N, n))
    elif distribution == "uniform":
        shifted_samples = np.random.uniform(low=params["low"], high=params["high"], size=(N, n))
        return shifted_samples + delta_mu
    elif distribution == "exponential":
        samples = np.random.exponential(scale=params["scale"], size=(N, n))
        return (samples - params["shift"]) + delta_mu
    else:
        raise ValueError("Неизвестное распределение")


# Проверка гипотезы
def hypothesis_one_sided_median(samples, n, a, alpha):
    """
    Проверяет гипотезу о том, что математическое ожидание меньше a.
    """
    means = np.mean(samples, axis=1)  # Выборочные средние
    stds = np.std(samples, axis=1, ddof=1)  # Выборочные стандартные отклонения
    t_crit = stats.t.ppf(1 - alpha, df=n - 1)  # t-критическое значение

    # Верхняя граница доверительного интервала
    upper_bounds = means + (stds / np.sqrt(n)) * t_crit

    # Отклоняем H0, если a < верхней границы интервала
    rejections = np.sum(a < upper_bounds)
    return rejections / N  # Частота отклонения гипотезы


# Сбор данных для графиков
results = {dist: {delta_mu: [] for delta_mu in delta_mu_values} for dist in
           ["normal", "uniform", "exponential"]}
ci_bounds = {dist: {delta_mu: [] for delta_mu in delta_mu_values} for dist in
             ["normal", "uniform", "exponential"]}

for n in n_values:
    for delta_mu in delta_mu_values:  # Теперь только для выбранных delta_mu
        for dist in ["normal", "uniform", "exponential"]:
            # Генерация выборок с учетом сдвига
            shifted_samples = generate_samples_shifted(
                n,
                dist,
                params_normal if dist == "normal" else params_uniform if dist == "uniform" else params_exponential,
                delta_mu
            )
            # Оценка alpha*
            alpha_star_val = hypothesis_one_sided_median(shifted_samples, n, a, alpha)
            results[dist][delta_mu].append(alpha_star_val)
            # Доверительный интервал для alpha*
            ci = 1.96 * np.sqrt((alpha_star_val * (1 - alpha_star_val)) / N)
            ci_bounds[dist][delta_mu].append((alpha_star_val - ci, alpha_star_val + ci))

# Построение графиков
plt.figure(figsize=(18, 10))
for dist in ["normal", "uniform", "exponential"]:
    plt.subplot(1, 3, ["normal", "uniform", "exponential"].index(dist) + 1)
    for delta_mu in delta_mu_values:  # Только для выбранных delta_mu
        alpha_vals = results[dist][delta_mu]
        ci_vals = ci_bounds[dist][delta_mu]
        lower_bounds, upper_bounds = zip(*ci_vals)

        # Построение кривой alpha*
        plt.plot(n_values, alpha_vals, label=f"Delta_mu={delta_mu:.2f}", marker='o')
        # Наложение доверительных интервалов
        plt.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2)

    plt.axhline(alpha, color="red", linestyle="--", label="Alpha (significance level)")
    plt.title(f"{dist.capitalize()} Distribution")
    plt.xlabel("Sample size (n)")
    plt.ylabel("Alpha*")
    plt.legend(loc="best", fontsize="small")
    plt.grid()

plt.tight_layout()
plt.show()
