import numpy as np
import matplotlib.pyplot as plt

def compute_ici_matrix(N, epsilon, alpha_D):
    """
    Быстрое вычисление матрицы I размера N x N.
    I[n,k] = exp(j*pi*z) * sinc(z), где z = (1+epsilon)*(n+1) + alpha_D - (k+1)
    """
    # Создаём индексы от 1 до N
    n = np.arange(1, N+1).reshape(-1, 1)   # столбец (N, 1)
    k = np.arange(1, N+1).reshape(1, -1)   # строка (1, N)
    
    # Вычисляем z для всей матрицы за один раз (broadcasting)
    z = (1 + epsilon) * n + alpha_D - k    # размер (N, N)
    
    # Вычисляем экспоненту и sinc
    I = np.exp(1j * np.pi * z) * np.sinc(z)
    return I

# ========================
# Параметры системы
# ========================
c = 3e8
v = 8000

N       = 2048                 # количество поднесущих
epsilon = v / c              # относительный сдвиг частоты дискретизации
alpha_D = 0#0.2              # нормализованный сдвиг несущей

# ========================
# 1. Генерация BPSK символов
# ========================
np.random.seed(42)         # для воспроизводимости
X = np.random.choice([-1, 1], size=N)   # переданные символы (±1)

I = compute_ici_matrix(N, epsilon, alpha_D)

# ========================
# 3. Формирование принятого сигнала Y[k]
#    Y[k] = sum_n H[n] * X[n] * I[n,k], H[n]=1 (плоский канал)
# ========================
Y = X @ I

# ========================
# 4. Демодуляция (BPSK: решение по знаку реальной части)
# ========================
X_est = np.sign(np.real(Y))      # восстанавливаем ±1
# для элементов, где real(Y) == 0, sign вернёт 0 – таких практически не бывает

# ========================
# 5. Подсчёт ошибок
# ========================
errors = np.sum(X != X_est)
ber = errors / N
print(f"Количество ошибочно демодулированных бит: {errors} из {N}")
print(f"BER = {ber:.4f}")

# ========================
# 6. Визуализация сигнальных созвездий
# ========================
plt.figure(figsize=(12, 5))

# Переданные символы
plt.subplot(1, 2, 1)
plt.scatter(np.real(X), np.imag(X), color='blue', s=50, label='Переданные')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(alpha=0.3)
plt.title('Созвездие на передатчике (BPSK)')
plt.xlabel('Re')
plt.ylabel('Im')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend()

# Принятые символы после канала (до демодулятора)
plt.subplot(1, 2, 2)
plt.scatter(np.real(Y), np.imag(Y), color='red', alpha=0.7, label='Принятые')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(alpha=0.3)
plt.title('Созвездие на приёмнике (до демодулятора)')
plt.xlabel('Re')
plt.ylabel('Im')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend()

plt.tight_layout()
plt.show()

# Дополнительно: отобразим несколько значений ICI для наглядности
print("\nПример элементов матрицы I (первые 5x5):")
for i in range(min(5, N)):
    for j in range(min(5, N)):
        print(f"I[{i+1},{j+1}] = {I[i,j]:.3f}")
    print()