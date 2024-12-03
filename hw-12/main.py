import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры
N = 1000  # количество выборок
n_values = [10, 50, 100, 150, 200, 500]  # размеры выборок
alpha = 0.05  # уровень значимости


# Функция генерации выборок из семейства распределений Пирсона
def generate_pearson_samples(n, N, kurtosis_target):
    samples = []
    for _ in range(N):
        beta_sample = np.random.beta(a=2, b=2, size=n)  # начальное распределение
        scaled_sample = (beta_sample - np.mean(beta_sample)) / np.std(beta_sample)  # нормализация
        adjusted_sample = scaled_sample * np.sqrt(kurtosis_target)  # регулировка эксцесса
        samples.append(adjusted_sample)
    return np.array(samples)


# Проверка гипотезы и вычисление долей отклонений
def perform_tests(samples, alpha):
    sw_rejections = []
    sf_rejections = []
    sample_kurtosis = []

    for sample in samples:
        # Расчет эксцесса
        kurt = stats.kurtosis(sample)
        sample_kurtosis.append(kurt)

        # Шапиро-Уилк
        sw_stat, sw_p = stats.shapiro(sample)
        sw_rejections.append(sw_p < alpha)

        # Шапиро-Франция через критерий Андерсона-Дарлинга
        sf_stat, critical_values, significance_levels = stats.anderson(sample, dist='norm')
        sf_rejections.append(sf_stat > critical_values[np.where(significance_levels == alpha * 100)[0][0]])

    return np.mean(sw_rejections), np.mean(sf_rejections), np.mean(sample_kurtosis)


# Генерация данных и тестирование
kurtosis_values = np.linspace(-1, 4, 6)  # значения эксцесса от -1 до 4
results = {'sw': [], 'sf': [], 'kurtosis': []}

for kurt in kurtosis_values:
    sw_results = []
    sf_results = []

    for n in n_values:
        samples = generate_pearson_samples(n, N, kurt)
        sw_rate, sf_rate, avg_kurtosis = perform_tests(samples, alpha)
        sw_results.append(sw_rate)
        sf_results.append(sf_rate)

    results['sw'].append(sw_results)
    results['sf'].append(sf_results)
    results['kurtosis'].append(avg_kurtosis)

# Визуализация результатов
plt.figure(figsize=(12, 8))
for i, kurt in enumerate(kurtosis_values):
    plt.plot(n_values, results['sw'][i], label=f"SW, эксцесс={kurt:.2f}", linestyle='-', color=f'C{i}')
    plt.plot(n_values, results['sf'][i], label=f"SF, эксцесс={kurt:.2f}", linestyle='--', color=f'C{i}')

plt.xlabel("Размер выборки (n)")
plt.ylabel("Доля отклонений нулевой гипотезы")
plt.title("Сравнение критериев Шапиро-Уилка и Шапиро-Франция")
plt.legend()
plt.grid()
plt.show()
