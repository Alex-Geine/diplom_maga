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

def theoretical_sinr(epsilon, N, SNR_dB=20):
    """
    Теоретический расчет SINR согласно предоставленным формулам
    """
    # Мощность сигнала
    P_signal = 1.0
    
    # Мощность шума
    sigma_n2 = P_signal / (10**(SNR_dB/10))
    
    sinr_per_subcarrier = np.zeros(N)
    
    for k in range(N):
        print(f"{k}/{N}")
        # Полезная составляющая: X[k]I[k,k]
        I_kk = calculate_I(k, k, epsilon)
        useful_power = P_signal * np.abs(I_kk)**2
        
        # Межканальная интерференция
        interference_power = 0
        for n in range(N):
            if n != k:
                I_nk = calculate_I(n, k, epsilon)
                interference_power += P_signal * np.abs(I_nk)**2
        
        # SINR в линейном масштабе
        sinr_linear = useful_power / (interference_power + sigma_n2)
        
        # В дБ
        sinr_per_subcarrier[k] = 10 * np.log10(sinr_linear)
    
    return sinr_per_subcarrier

# Пример использования
N = 2048
SNR_dB = 30

max_eps = 8e3 / 3e8

# Различные значения доплеровского коэффициента
persents = np.linspace(0.1, 1, 5)
epsilons = max_eps * persents

plt.figure(figsize=(12, 7))

for i  in range(len(persents)):
    sinr = theoretical_sinr(epsilons[i], N, SNR_dB)
    plt.plot(range(N), sinr, 
             label=f'ε = {persents[i]} * ε_max', 
             linewidth=2, 
             marker='o' if len(epsilons) <= 3 else None,
             markersize=4)

plt.xlabel('Индекс поднесущей (k)', fontsize=12)
plt.ylabel('SINR, дБ', fontsize=12)
plt.title(f'Теоретический SINR для OFDM системы\n(N={N}, SNR={SNR_dB} дБ)', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
