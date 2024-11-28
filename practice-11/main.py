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


# Пункт 1
# Функция для генерации выборок
def generate_samples(n, distribution, params, delta_mu=0):
    if distribution == "normal":
        return np.random.normal(loc=params["mean"] + delta_mu, scale=params["std"], size=(N, n))
    elif distribution == "uniform":
        samples = np.random.uniform(low=params["low"], high=params["high"], size=(N, n))
        return samples + delta_mu
    elif distribution == "exponential":
        samples = np.random.exponential(scale=params["scale"], size=(N, n))
        return (samples - params["shift"]) + delta_mu


# Проверка гипотезы
def hypothesis_one_sided_median(samples, n, a, alpha):
    means = np.mean(samples, axis=1)
    stds = np.std(samples, axis=1, ddof=1)
    t_crit = stats.t.ppf(1 - alpha, df=n - 1)
    upper_bounds = means + (stds / np.sqrt(n)) * t_crit
    rejections = np.sum(a < upper_bounds)
    return rejections / N


# Сбор данных
alpha_star_n = {dist: [] for dist in ["normal", "uniform", "exponential"]}
ci_bounds_n = {dist: [] for dist in ["normal", "uniform", "exponential"]}
results_mu = {dist: {delta_mu: [] for delta_mu in delta_mu_values} for dist in ["normal", "uniform", "exponential"]}
ci_bounds_mu = {dist: {delta_mu: [] for delta_mu in delta_mu_values} for dist in ["normal", "uniform", "exponential"]}

for n in n_values:
    for dist in ["normal", "uniform", "exponential"]:
        # Пункт 2: фиксированный delta_mu = 0
        samples = generate_samples(n, dist,
                                   params_normal if dist == "normal" else params_uniform if dist == "uniform" else params_exponential)
        alpha_star_val = hypothesis_one_sided_median(samples, n, a, alpha)
        alpha_star_n[dist].append(alpha_star_val)
        ci = 1.96 * np.sqrt((alpha_star_val * (1 - alpha_star_val)) / N)
        ci_bounds_n[dist].append((alpha_star_val - ci, alpha_star_val + ci))

        # Пункт 3: различные delta_mu
        for delta_mu in delta_mu_values:
            shifted_samples = generate_samples(n, dist,
                                               params_normal if dist == "normal" else params_uniform if dist == "uniform" else params_exponential,
                                               delta_mu)
            alpha_star_val = hypothesis_one_sided_median(shifted_samples, n, a, alpha)
            results_mu[dist][delta_mu].append(alpha_star_val)
            ci = 1.96 * np.sqrt((alpha_star_val * (1 - alpha_star_val)) / N)
            ci_bounds_mu[dist][delta_mu].append((alpha_star_val - ci, alpha_star_val + ci))

# Построение графиков

# Пункт 2, график: зависимость alpha* от n
plt.figure(figsize=(10, 6))
for dist in ["normal", "uniform", "exponential"]:
    alpha_vals = alpha_star_n[dist]
    ci_vals = ci_bounds_n[dist]
    lower_bounds, upper_bounds = zip(*ci_vals)
    plt.plot(n_values, alpha_vals, label=f"{dist.capitalize()} (alpha*)", marker='o')
    plt.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2)

plt.axhline(alpha, color="red", linestyle="--", label="Уровень значимости alpha")
plt.title("Зависимость alpha* от размера выборки (n)")
plt.xlabel("Размер выборки (n)")
plt.ylabel("alpha*")
plt.legend()
plt.grid()
plt.show()

# Пункт 3, график: зависимость alpha* от n при разных delta_mu
plt.figure(figsize=(24, 8))
handles, labels = [], []  # Переменные для сохранения элементов легенды

# Устанавливаем шаг для delta_mu в легенде
legend_step = 0.2

# Получаем наибольшее и наименьшее значение delta_mu
min_delta_mu = min(delta_mu_values)
max_delta_mu = max(delta_mu_values)

# Создаем список уникальных delta_mu для легенды
selected_delta_mu = set()

for dist in ["normal", "uniform", "exponential"]:
    plt.subplot(1, 3, ["normal", "uniform", "exponential"].index(dist) + 1)
    for delta_mu in delta_mu_values:
        alpha_vals = results_mu[dist][delta_mu]
        ci_vals = ci_bounds_mu[dist][delta_mu]
        lower_bounds, upper_bounds = zip(*ci_vals)

        # Построение кривой alpha*
        line, = plt.plot(n_values, alpha_vals, label=f"Delta_mu={delta_mu:.2f}", marker='o')
        # Наложение доверительных интервалов
        plt.fill_between(n_values, lower_bounds, upper_bounds, alpha=0.2)

        # Добавляем в легенду значения delta_mu с шагом и крайние значения
        if (
                abs((delta_mu - min_delta_mu) % legend_step) < 1e-6 or  # Кратность шагу
                delta_mu == min_delta_mu or  # Наименьшее значение
                delta_mu == max_delta_mu  # Наибольшее значение
        ) and delta_mu not in selected_delta_mu:  # Исключаем повторения
            handles.append(line)
            labels.append(f"Delta_mu={delta_mu:.2f}")
            selected_delta_mu.add(delta_mu)

    plt.axhline(alpha, color="red", linestyle="--", label="Alpha (significance level)")
    plt.title(f"{dist.capitalize()} Distribution")
    plt.xlabel("Sample size (n)")
    plt.ylabel("Alpha*")
    plt.grid()

# Добавление общей легенды с выбранными значениями delta_mu
plt.legend(handles, labels + ["Alpha (significance level)"], loc="center left", fontsize="small", title="Delta_mu",
           bbox_to_anchor=(1.02, 0.5), ncol=1)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Увеличиваем отступ для легенды
plt.suptitle("Зависимость alpha* от размера выборки при разных delta_mu", fontsize=16)
plt.show()
