import numpy as np
import pytest
from scipy.stats import chi2, norm

import main


def float_to_bits(samples):
    # Преобразование массива чисел с плавающей запятой в двоичную строку фиксированной длины 32 бита
    # Умножаем на 2^32, чтобы получить целочисленное представление, затем форматируем в двоичную строку
    return ''.join(format(int(sample * (2 ** 32)), '032b') for sample in samples)


def frequency_bit_test(bit_string):
    """  Реализация теста частоты, который проверяет равновесие между нулями и единицами """
    # Определяем длину битовой строки
    n = len(bit_string)

    # Считаем количество единиц в битовой строке
    count_ones = bit_string.count('1')

    # Ожидаемое количество единиц и нулей
    # Если строка длиной n, то при равновесии ожидается, что половина - единицы, а половина - нули
    expected_count = n / 2

    # Формула Z-статистики для проверки гипотезы о равновесии между 0 и 1
    # Z = (X - μ) / σ, где:
    # X - фактическое количество единиц (count_ones)
    # μ - ожидаемое количество единиц (expected_count)
    # σ - стандартное отклонение для бинарного распределения
    # Для бинарного распределения с параметрами p=0.5: σ = sqrt(n * p * (1 - p)) = sqrt(n * 0.5 * 0.5) = sqrt(n) / 2
    z_frequency = (count_ones - expected_count) / (expected_count * (1 / 2) ** 0.5)

    # Вычисляем p-value из нормального распределения
    # Используем функцию кумулятивного распределения (CDF) нормального распределения
    # P(Z > z_frequency) = 1 - CDF(z_frequency)
    p_value_frequency = 1 - norm.cdf(z_frequency)

    return p_value_frequency


def block_frequency_test(bit_string, m):
    """ Реализация блочного теста частоты, который проверяет равномерность распределения единиц в блоках """
    n = len(bit_string)  # Длина двоичной строки
    k = n // m  # Количество блоков
    blocks = [bit_string[i * m:(i + 1) * m] for i in range(k)]  # Разделение на блоки

    # Подсчет единиц в каждом блоке
    ones_count = [block.count('1') for block in blocks]

    # Ожидаемое количество единиц и дисперсия для каждого блока
    expected = m / 2
    variance = m / 4

    # Статистика хи-квадрат вычисляется как сумма квадратов отклонений наблюдаемых значений от ожидаемых,
    # нормированных на дисперсию
    csi_2 = sum(((count - expected) ** 2) / variance for count in ones_count)

    # chi2.cdf вычисляет кумулятивную функцию распределения для хи-квадрат,
    # которая возвращает вероятность того, что случайная величина с распределением хи-квадрат примет значение <= csi_2
    p_value = 1 - chi2.cdf(csi_2, k - 1)
    return p_value


def consecutive_bits_test(bit_string):
    """ Реализация теста на одинаковые подряд идущие биты """
    n = len(bit_string)  # Длина двоичной строки
    max_run_length = 0  # Максимальная длина последовательности
    current_run_length = 1  # Текущая длина последовательности

    # Подсчет длины последовательностей
    for i in range(1, n):
        # Если текущий бит равен предыдущему, увеличиваем текущую длину последовательности
        if bit_string[i] == bit_string[i - 1]:
            current_run_length += 1
        else:
            max_run_length = max(max_run_length, current_run_length)
            current_run_length = 1

    # Проверка последней последовательности
    max_run_length = max(max_run_length, current_run_length)

    # Формула для ожидаемого количества последовательностей (runs):
    # E(R) = (n - L + 1) / (2 * L), где L - максимальная длина последовательности
    expected_runs = (n - max_run_length + 1) / (2 ** max_run_length)

    # Формула для дисперсии:
    # Var(R) = (n - L + 1) * (1 / (2 * L)) * (1 - (1 / (2 ** L)))
    variance = (n - max_run_length + 1) * (1 / (2 ** max_run_length)) * (1 - (1 / (2 ** max_run_length)))

    # Z-статистика позволяет оценить отклонение наблюдаемого значения от ожидаемого,
    # нормированного на стандартное отклонение. Если дисперсия равна 0, Z-статистика равна 0.
    z = (max_run_length - expected_runs) / np.sqrt(variance) if variance > 0 else 0

    # p-value из нормального распределения
    p_value = 1 - norm.cdf(z)
    return p_value


def longest_run_test(bit_string, block_size):
    """ Реализация теста на самую длинную последовательность единиц в блоке """
    # Определяем длину битовой строки
    n = len(bit_string)

    # Количество блоков, на которые будет разбита битовая строка
    num_blocks = n // block_size
    # Список для хранения максимальной длины последовательности единиц в каждом блоке
    max_runs = []

    for i in range(num_blocks):
        # Извлекаем текущий блок из битовой строки
        block = bit_string[i * block_size:(i + 1) * block_size]
        current_run_length = 0  # Текущая длина последовательности единиц
        max_run_length = 0  # Максимальная длина последовательности единиц в текущем блоке

        for bit in block:
            if bit == '1':
                current_run_length += 1
                # Обновляем максимальную длину последовательности, если текущая больше
                max_run_length = max(max_run_length, current_run_length)
            else:
                # Если встретили '0', сбрасываем текущую длину последовательности
                current_run_length = 0

        # Добавляем максимальную длину последовательности единиц из текущего блока в список
        max_runs.append(max_run_length)

    # Ожидаемое значение максимальной длины последовательности единиц
    expected_max_run_length = (block_size + 1) / 2

    # Дисперсия для максимальной длины последовательности единиц
    # Формула: Var(X) = (n/2) * (1 - p), где p = вероятность появления '1'
    variance = (block_size / 2) * (1 - (1 / 2))

    # Формула Z: Z = (X̄ - μ) / σ, где X̄ - среднее значение выборки,
    # μ - ожидаемое значение, σ - стандартное отклонение (sqrt(Var))
    z = (sum(max_runs) / num_blocks - expected_max_run_length) / (variance ** 0.5)

    # Вычисляем p-value из нормального распределения
    # P(Z > z) = 1 - CDF(z)
    p_value = 1 - norm.cdf(z)
    return p_value


def rank_test(bit_string, rows, cols):
    """ Реализация теста рангов для бинарной матрицы. """
    # Удаляем все символы, кроме '0' и '1'
    bit_string = ''.join(filter(lambda x: x in '01', bit_string))

    # Разбиваем строку на подстроки длиной cols и создаем матрицу размером rows x cols
    matrix = np.array([list(bit_string[i * cols:(i + 1) * cols]) for i in range(rows)], dtype=int)

    # Находим ранг матрицы
    rank = np.linalg.matrix_rank(matrix)

    # Ожидаемое значение ранга для случайной бинарной матрицы размером rows x cols
    expected_rank = min(rows, cols)

    # Дисперсия рассчитывается для оценки разброса значений ранга
    # Формула: Var(X) = (rows * cols) / 2 * (1 - p), где p = вероятность появления '1'
    variance = (rows * cols) / 2 * (1 - (1 / 2))

    # Стандартизируем разницу между наблюдаемым рангом и ожидаемым значением
    # Формула: Z = (X̄ - μ) / σ, где X̄ - наблюдаемый ранг, μ - ожидаемое значение ранга,
    # σ - стандартное отклонение (квадратный корень из дисперсии)
    z = (rank - expected_rank) / (variance ** 0.5)

    # Вычисляем p-value из нормального распределения
    # P(Z > z) = 1 - CDF(z)
    p_value = 1 - norm.cdf(z)
    return p_value


def spectral_test(bit_string):
    """ Реализация спектрального теста """
    # Удаляем все символы, кроме '0' и '1'
    bit_string = ''.join(filter(lambda x: x in '01', bit_string))
    n = len(bit_string)

    # Используем map для преобразования каждого символа строки в целое число (0 или 1)
    bits = np.array(list(map(int, bit_string)))

    # Вычисляем дискретное преобразование Фурье
    fft_result = np.fft.fft(bits)

    # Амплитуда — это модуль комплексных чисел, полученных в результате FFT
    amplitudes = np.abs(fft_result)

    # Получаем среднее значение амплитуд
    mean_amplitude = np.mean(amplitudes)

    # Ожидаемое значение амплитуды для случайной последовательности
    expected_amplitude = np.sqrt(n)

    # Стандартизируем разницу между наблюдаемым средним значением амплитуды и ожидаемым значением
    # Формула: Z = (X̄ - μ) / (σ / √n), где X̄ - наблюдаемое среднее, μ - ожидаемое значение,
    # σ - стандартное отклонение, которое в данном случае принимается равным ожидаемой амплитуде
    z_spectral = (mean_amplitude - expected_amplitude) / (expected_amplitude / np.sqrt(2 * n))

    # Вычисляем p-value из нормального распределения
    # P(Z > z) = 1 - CDF(z)
    p_value = 1 - norm.cdf(z_spectral)
    return p_value


def run_randomness_test(samples):
    # Выполнение тестов
    bit_string = float_to_bits(samples)  # Преобразование выборки в двоичную строку

    # Запуск теста частоты
    p_value_frequency = frequency_bit_test(bit_string)
    print(f'\n Frequency Bit Test p-value: {p_value_frequency}')

    # Запуск блочного теста частоты с длиной блока m
    m = 10  # Длина блока
    p_value_block = block_frequency_test(bit_string, m)
    print(f'\n Block Frequency Test p-value: {p_value_block}')

    # Запуск теста на самую длинную последовательность единиц в блоке
    p_value_longest_run = longest_run_test(bit_string, m)
    print(f'\n Longest Run Test p-value: {p_value_longest_run}')

    # Запуск теста рангов бинарной матрицы
    rows, cols = 5, 5  # Размеры матрицы
    p_value_rank = rank_test(bit_string, rows, cols)
    print(f'\n Rank Test p-value: {p_value_rank}')

    # Запуск спектрального теста
    p_value_spectral = spectral_test(bit_string)
    print(f'\n Spectral Test p-value: {p_value_spectral}')

    return p_value_frequency, p_value_block, p_value_longest_run, p_value_rank, p_value_spectral


# Тестовая функция для проверки случайности различных распределений
@pytest.mark.parametrize("distribution_name",
                         ["uniform_0_1", "uniform_neg1_1", "sum_of_three_uniforms", "cauchy_distribution"])
def test_run_randomness(distribution_name):
    N = 1000  # Количество образцов для генерации
    distributions = main.generate_samples(N)  # Генерация образцов из разных распределений

    # Получаем соответствующее распределение по имени
    samples = {
        "uniform_0_1": distributions[0],
        "uniform_neg1_1": distributions[1],
        "sum_of_three_uniforms": distributions[2],
        "cauchy_distribution": distributions[3]
    }[distribution_name]

    # Обработка выбросов для распределения Коши
    if distribution_name == "cauchy_distribution":
        samples = np.clip(samples, -1000, 1000)  # Ограничиваем значения, чтобы избежать влияния выбросов

    # Выполнение тестов
    p_values = run_randomness_test(samples)

    # Проверка корректности полученных p-values
    for p_value in p_values:
        assert 0 <= p_value <= 1, f"\n p-value должен быть в пределах от 0 до 1 для {distribution_name}"
        assert p_value > 0.01, f"\n p-value слишком мал для {distribution_name}, это может указывать на неслучайность"
