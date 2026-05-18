import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Функции модуляции и демодуляции (5G NR)
# ============================================

def get_constellation(modulation):
    """
    Возвращает список комплексных символов созвездия, нормированных на среднюю мощность 1.
    Порядок символов соответствует стандартному отображению 5G NR (Gray mapping).
    """
    if modulation == 'BPSK':
        # BPSK: ±1
        symbols = np.array([-1, 1])
        # мощность уже 1
    elif modulation == 'QPSK':
        # QPSK: (±1 ± i) / sqrt(2)
        symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif modulation == '16QAM':
        # 16QAM: амплитуды ±1, ±3, нормировка на sqrt(10)
        amplitudes = np.array([-3, -1, 1, 3])
        symbols = (amplitudes[:, None] + 1j*amplitudes[None, :]).flatten()
        symbols = symbols / np.sqrt(10)   # средняя мощность 1
    elif modulation == '64QAM':
        # 64QAM: амплитуды ±1, ±3, ±5, ±7, нормировка на sqrt(42)
        amplitudes = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        symbols = (amplitudes[:, None] + 1j*amplitudes[None, :]).flatten()
        symbols = symbols / np.sqrt(42)
    elif modulation == '256QAM':
        # 256QAM: амплитуды ±1,±3,±5,±7,±9,±11,±13,±15, нормировка на sqrt(170)
        amplitudes = np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
        symbols = (amplitudes[:, None] + 1j*amplitudes[None, :]).flatten()
        symbols = symbols / np.sqrt(170)
    else:
        raise ValueError(f"Неизвестная модуляция: {modulation}")
    return symbols

def bits_to_symbols(bits, modulation):
    """
    Преобразует последовательность битов в комплексные символы.
    bits: одномерный массив битов (0/1)
    возвращает: массив комплексных символов
    """
    symbols = get_constellation(modulation)
    M = len(symbols)          # размер созвездия
    bps = int(np.log2(M))     # бит на символ
    if len(bits) % bps != 0:
        raise ValueError("Количество битов не кратно bps")
    # Группируем биты в целые числа (индексы)
    bits_grouped = bits.reshape(-1, bps)
    indices = np.zeros(len(bits_grouped), dtype=int)
    for i in range(bps):
        indices += bits_grouped[:, i] << (bps - 1 - i)  # старший бит первым (MSB first)
    return symbols[indices]

def symbols_to_bits(symbols_rx, modulation):
    """
    Демодуляция (жесткое решение) – поиск ближайшего символа из созвездия.
    symbols_rx: принятые комплексные символы (искажённые)
    возвращает: массив битов (0/1)
    """
    const = get_constellation(modulation)
    bps = int(np.log2(len(const)))
    bits = []
    for sym in symbols_rx:
        # евклидово расстояние до всех символов созвездия
        dists = np.abs(const - sym)
        nearest_idx = np.argmin(dists)
        # преобразуем индекс в биты (MSB first)
        for i in range(bps):
            bit = (nearest_idx >> (bps - 1 - i)) & 1
            bits.append(bit)
    return np.array(bits)

# ============================================
# Быстрое вычисление матрицы интерференции I
# ============================================
def compute_ici_matrix(N, epsilon, alpha_D):
    """
    I[n,k] = exp(j*pi*z) * sinc(z)
    z = (1+epsilon)*(n+1) + alpha_D - (k+1)
    n, k = 1..N
    """
    n = np.arange(1, N+1).reshape(-1, 1)   # столбец (N,1)
    k = np.arange(1, N+1).reshape(1, -1)   # строка (1,N)
    z = (1 + epsilon) * n + alpha_D - k
    I = np.exp(1j * np.pi * z) * np.sinc(z)
    return I

# ============================================
# Параметры системы
# ============================================
c = 3e8
v = 8000
epsilon = v / c               # относительный сдвиг
alpha_D = 0.0                 # сдвиг несущей

N = 4096                      # количество поднесущих
modulation = '256QAM'          # выберите: 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'

# Генерация случайных битов (достаточно для N символов)
constellation = get_constellation(modulation)
bps = int(np.log2(len(constellation)))   # бит на символ
num_bits_total = N * bps
np.random.seed(42)
bits_tx = np.random.randint(0, 2, size=num_bits_total)

# Отображение битов в символы
X = bits_to_symbols(bits_tx, modulation)   # переданные символы (N штук)

# Матрица интерференции
I = compute_ici_matrix(N, epsilon, alpha_D)

# Принятый сигнал (плоский канал H=1)
Y = X @ I

# Демодуляция принятых символов в биты
bits_rx = symbols_to_bits(Y, modulation)

# Подсчёт ошибочных битов
errors = np.sum(bits_tx != bits_rx)
ber = errors / num_bits_total
print(f"Модуляция: {modulation}")
print(f"Всего битов: {num_bits_total}")
print(f"Ошибочных битов: {errors}")
print(f"BER = {ber:.6f}")

# ============================================
# Визуализация созвездий
# ============================================
plt.figure(figsize=(12, 5))

# Переданные символы (идеальное созвездие)
plt.subplot(1, 2, 1)
plt.scatter(np.real(X), np.imag(X), c='blue', s=10, alpha=0.6, label='Переданные')
# Границы для наглядности
max_amp = max(np.abs(X).max(), np.abs(Y).max()) + 0.5
plt.xlim(-max_amp, max_amp)
plt.ylim(-max_amp, max_amp)
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(alpha=0.3)
plt.title(f'Созвездие передатчика ({modulation})')
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()

# Принятые символы (после ICI)
plt.subplot(1, 2, 2)
plt.scatter(np.real(Y), np.imag(Y), c='red', s=10, alpha=0.6, label='Принятые')
plt.xlim(-max_amp, max_amp)
plt.ylim(-max_amp, max_amp)
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(alpha=0.3)
plt.title(f'Созвездие приёмника (ICI)')
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()

plt.tight_layout()
plt.show()

# Несколько элементов матрицы I для проверки
print("\nПример элементов матрицы I (первые 5x5):")
for i in range(min(5, N)):
    for j in range(min(5, N)):
        print(f"I[{i+1},{j+1}] = {I[i,j]:.3f}")
    print()