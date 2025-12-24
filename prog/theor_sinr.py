import numpy as np
import matplotlib.pyplot as plt

def calculate_I(n, k, epsilon):
    """
    Расчет коэффициента I[n, k] из формулы
    I[n, k] = exp(jπ[(1+ε)n - k]) * sinc(π[(1+ε)n - k])
    """
    arg = np.pi * ((1 + epsilon) * n - k)
    
    # Для sinc(πx) нужно использовать sin(πx)/(πx)
    if np.abs(arg) < 1e-10:  # Избегаем деления на 0
        sinc_val = 1.0
    else:
        sinc_val = np.sin(arg) / arg
    
    # Экспоненциальный множитель
    exp_val = np.exp(1j * arg)
    
    return exp_val * sinc_val

def theoretical_sinr(epsilon, N, ids, SNR_dB=20):
    P_signal = 1.0
    sigma_n2 = P_signal / (10**(SNR_dB/10))
    
    # Создаем массив той же длины, что и набор индексов
    sinr_per_subcarrier = np.zeros(len(ids))
    
    # Используем enumerate, чтобы k_val было реальным индексом (0, 10, 20...),
    # а idx — порядковым номером в массиве результатов (0, 1, 2...)
    for idx, k_val in enumerate(ids):
        # Полезная составляющая: используем k_val
        I_kk = calculate_I(k_val, k_val, epsilon)
        useful_power = P_signal * np.abs(I_kk)**2
        
        interference_power = 0
        for n in range(N):
            if n != k_val:
                I_nk = calculate_I(n, k_val, epsilon)
                interference_power += P_signal * np.abs(I_nk)**2
        
        sinr_linear = useful_power / (interference_power + sigma_n2)
        sinr_per_subcarrier[idx] = 10 * np.log10(sinr_linear)
        
    return sinr_per_subcarrier

# Пример использования
N = 2048
SNR_dB = 20

vel = np.array([100, 1000, 4000, 8000])

# Различные значения доплеровского коэффициента
epsilons = vel / 3e8

plt.figure(figsize=(12, 7))

sinrs = [None] * len(vel)

ids = range(0, N, 50)

for i  in range(len(vel)):
    sinrs[i] = theoretical_sinr(epsilons[i], N, ids, SNR_dB)

plt.plot(ids, sinrs[0], 
            color='black', linestyle='-', marker='^', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Теоретическая кривая (V = 100 м/c)')

plt.plot(ids, sinrs[1], 
            color='black', linestyle='-', marker='s', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Теоретическая кривая (V = 1000 м/c)')

plt.plot(ids, sinrs[2], 
            color='black', linestyle='-', marker='o', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Теоретическая кривая (V = 4000 м/c)')

plt.plot(ids, sinrs[3], 
            color='black', linestyle='-', marker='d', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Теоретическая кривая (V = 8000 м/c)')

plt.xlabel('Индекс поднесущей (k)', fontsize=24)
plt.ylabel('SINR, дБ', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)  # Основные деления
plt.tick_params(axis='both', which='minor', labelsize=16)  # Промежуточные (мелкие) деления
#plt.title(f'Теоретический SINR для OFDM системы\n(N={N}, SNR={SNR_dB} дБ)', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')

plt.legend(fontsize=18)
plt.tight_layout()
#plt.show()
filename = f'theor_sinr.png'
filenamePdf = f'theor_sinr.pdf'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.savefig(filenamePdf, dpi=300, bbox_inches='tight')
