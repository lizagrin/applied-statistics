import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size)


def generate_normal(size):
    return np.random.normal(0, 1, size)


def generate_sum_2_uniform_minus1_1(size):
    return np.random.uniform(-1, 1, size) + np.random.uniform(-1, 1, size)


def mad(data):
    # Вычисляет среднее абсолютное отклонение (MAD) для заданных данных
    median = np.median(data)  # Находим медиану данных
    return np.mean(np.abs(data - median))  # Вычисляем среднее абсолютное отклонение от медианы


def mad_confidence_interval(data, confidence_level=0.95):
    # Вычисляет доверительный интервал для среднего абсолютного отклонения (MAD)
    n = len(data)  # Размер выборки
    mad_value = mad(data)  # Вычисляем MAD
    sigma_est = mad_value / 0.67449  # Оценка стандартного отклонения на основе MAD
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)  # Z-значение для заданного уровня доверия
    half_width = z_value * sigma_est / np.sqrt(n)  # Половина ширины доверительного интервала
    return mad_value - half_width, mad_value + half_width


def bootstrap_mad(data, n_bootstrap=1000, confidence_level=0.95):
    # Вычисляет среднее значение и доверительный интервал для MAD с использованием бутстрепа
    bootstraps = np.random.choice(data, (n_bootstrap, len(data)), replace=True)  # Генерируем бутстреп-выборки
    mad_values = np.array([mad(sample) for sample in bootstraps])  # Вычисляем MAD для каждой бутстреп-выборки
    lower_bound = np.percentile(mad_values, (1 - confidence_level) / 2 * 100)  # Нижняя граница интервала
    upper_bound = np.percentile(mad_values, (1 + confidence_level) / 2 * 100)  # Верхняя граница интервала
    return np.mean(mad_values), lower_bound, upper_bound


def jackknife_mad(data):
    # Вычисляет среднее значение и стандартное отклонение для MAD с использованием метода джекнайфа
    n = len(data)
    mad_values = np.array([mad(np.delete(data, i)) for i in range(n)])
    jackknife_mean = np.mean(mad_values)
    jackknife_var = (n - 1) * np.mean((mad_values - jackknife_mean) ** 2)
    jackknife_std = np.sqrt(jackknife_var)
    return jackknife_mean, jackknife_std


# Размеры выборок
sample_sizes = [500, 1000, 5000, 10000]

# Определение распределений
distributions = {
    "Uniform (-1, 1)": generate_uniform_minus1_1,
    "Normal (0, 1)": generate_normal,
    "Sum of two Uniform (-1, 1)": generate_sum_2_uniform_minus1_1
}

# Построение графиков для каждого распределения
for dist_name, dist_func in distributions.items():
    print(f"\n{dist_name}")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Инициализация списков для хранения результатов
    mad_values = []
    ci_lower = []
    ci_upper = []

    boot_mad_values = []
    boot_ci_lower = []
    boot_ci_upper = []

    jack_mad_values = []
    jack_ci_lower = []
    jack_ci_upper = []

    for size in sample_sizes:
        data = dist_func(size)

        # Вычисление MAD и доверительных интервалов
        mad_val = mad(data)
        ci = mad_confidence_interval(data)

        boot_mad_val, boot_ci_lower_val, boot_ci_upper_val = bootstrap_mad(data)

        jack_mad_val, jack_std_val = jackknife_mad(data)
        # Используем нормальное распределение для расчета доверительных интервалов
        z_score = 1.96  # для 95% доверительного интервала
        jack_ci_val = (jack_mad_val - z_score * jack_std_val, jack_mad_val + z_score * jack_std_val)

        # Сохранение результатов
        mad_values.append(mad_val)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])

        boot_mad_values.append(boot_mad_val)
        boot_ci_lower.append(boot_ci_lower_val)
        boot_ci_upper.append(boot_ci_upper_val)

        jack_mad_values.append(jack_mad_val)
        jack_ci_lower.append(jack_ci_val[0])
        jack_ci_upper.append(jack_ci_val[1])

        # Вывод результатов
        print(f"Sample Size: {size}")
        print(f"  MAD: {mad_val:.4f}, CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Bootstrap MAD: {boot_mad_val:.4f}, CI: [{boot_ci_lower_val:.4f}, {boot_ci_upper_val:.4f}]")
        print(f"  Jackknife MAD: {jack_mad_val:.4f}, CI: [{jack_ci_val[0]:.4f}, {jack_ci_val[1]:.4f}]\n")

    # Отображение результатов в виде точек
    x_positions = np.arange(len(sample_sizes))
    width = 0.25

    # Отображение точек MAD
    ax.scatter(x_positions - width, mad_values, label='MAD', color='skyblue', zorder=3)
    ax.scatter(x_positions, boot_mad_values, label='Bootstrap MAD', color='salmon', zorder=3)
    ax.scatter(x_positions + width, jack_mad_values, label='Jackknife MAD', color='lightgreen', zorder=3)

    # Добавление доверительных интервалов
    for x_pos, mad_val, ci_lower_val, ci_upper_val in zip(x_positions - width, mad_values, ci_lower, ci_upper):
        ax.errorbar(x_pos, mad_val,
                    yerr=[[mad_val - ci_lower_val], [ci_upper_val - mad_val]],
                    fmt='o', color='black', capsize=5, zorder=2)

    for x_pos, boot_mad_val, boot_ci_lower_val, boot_ci_upper_val in zip(x_positions, boot_mad_values, boot_ci_lower,
                                                                         boot_ci_upper):
        ax.errorbar(x_pos, boot_mad_val,
                    yerr=[[boot_mad_val - boot_ci_lower_val], [boot_ci_upper_val - boot_mad_val]],
                    fmt='o', color='black', capsize=5, zorder=2)

    for x_pos, jack_mad_val, jack_ci_lower_val, jack_ci_upper_val in zip(x_positions + width, jack_mad_values,
                                                                         jack_ci_lower, jack_ci_upper):
        ax.errorbar(x_pos, jack_mad_val,
                    yerr=[[jack_mad_val - jack_ci_lower_val], [jack_ci_upper_val - jack_mad_val]],
                    fmt='o', color='black', capsize=5, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sample_sizes)  # Устанавливаем метки оси X в соответствии с размерами выборок
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('MAD')
    ax.set_title(f'MAD Confidence Intervals for {dist_name}')
    ax.legend()

    plt.grid(True, linestyle='--', alpha=0.7)  # Добавление сетки для лучшей читаемости
    plt.show()
