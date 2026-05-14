import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def scale_spectrum_ofdm(X_fd, a):
    """
    Масштабирует спектр X(f) -> X(af) с сохранением сетки N и нормировкой энергии.
    """
    if a <= 0:
        raise ValueError("Коэффициент 'a' должен быть больше нуля.")
        
    N = len(X_fd)
    
    # 1. Центрирование спектра для корректного изменения масштаба вокруг DC
    X_shifted = np.fft.fftshift(X_fd)
    
    # 2. Изменение разрешения спектра
    new_N = int(round(N / a))
    if new_N <= 0:
        raise ValueError("Коэффициент 'a' слишком велик.")
    X_resampled = resample(X_shifted, new_N)
    
    # 3. Возврат к исходному размеру N (Zero-padding или Трункация)
    X_final_shifted = np.zeros(N, dtype=complex)
    if new_N >= N:
        # Растяжение спектра (a < 1) -> отсечение краев
        start = (new_N - N) // 2
        X_final_shifted = X_resampled[start:start + N]
    else:
        # Сжатие спектра (a > 1) -> заполнение краев нулями
        start = (N - new_N) // 2
        X_final_shifted[start:start + new_N] = X_resampled
        
    # 4. Восстановление стандартного порядка частот FFT
    X_scaled = np.fft.ifftshift(X_final_shifted)
    
    # 5. Нормировка амплитуды для сохранения энергии исходного сигнала
    # Энергия дискретного сигнала в частотной области: E = sum(|X|^2) / N
    # energy_initial = np.sum(np.abs(X_fd)**2)
    # energy_current = np.sum(np.abs(X_scaled)**2)
    
    # if energy_current > 0:
        # X_scaled = X_scaled * np.sqrt(energy_initial / energy_current)
        
    return X_scaled

# --- Симуляция OFDM QPSK сигнала ---
N =1048  # Количество поднесущих
np.random.seed(42)

# Генерация случайных символов QPSK (созвездие: (+-1 +- j)/sqrt(2))
qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
symbols = np.random.choice(qpsk_constellation, N)

# В OFDM спектр формируется непосредственно в частотной области
X_initial = symbols.copy()

# Масштабирование спектра (например, сжатие в 1.5 раза)
a = 100
X_scaled = scale_spectrum_ofdm(X_initial, a)

# --- Визуализация спектров ---
# Частотная сетка (индексы поднесущих после fftshift)
freq_bins = np.fft.fftshift(np.fft.fftfreq(N, d=1))

plt.figure(figsize=(12, 5))

# График исходного спектра
plt.subplot(1, 2, 1)
plt.stem(freq_bins, np.abs(np.fft.fftshift(X_initial)), basefmt=" ", linefmt="C0-", markerfmt="C0o")
plt.title("Исходный спектр OFDM (QPSK)")
plt.xlabel("Нормированная частота")
plt.ylabel("Амплитуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0, 1.4)

# График масштабированного спектра
plt.subplot(1, 2, 2)
plt.stem(freq_bins, np.abs(np.fft.fftshift(X_scaled)), basefmt=" ", linefmt="C1-", markerfmt="C1o")
plt.title(f"Масштабированный спектр (a = {a})")
plt.xlabel("Нормированная частота")
plt.ylabel("Амплитуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0, 1.4)

plt.tight_layout()
plt.show()

# Проверка сохранения энергии
print(f"Энергия исходного спектра: {np.sum(np.abs(X_initial)**2):.4f}")
print(f"Энергия масштабированного спектра: {np.sum(np.abs(X_scaled)**2):.4f}")
