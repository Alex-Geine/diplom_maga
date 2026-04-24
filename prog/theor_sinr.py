import numpy as np
import matplotlib.pyplot as plt

def calculate_I(n, k, epsilon, aD):
    """
    Расчет коэффициента I[n, k] из формулы
    I[n, k] = exp(jπ[(1+ε)n + aD - k]) * sinc(π[(1+ε)n + aD - k])
    """
    arg = np.pi * ((1 + epsilon) * n + aD - k)
    
    # Для sinc(πx) нужно использовать sin(πx)/(πx)
    if np.abs(arg) < 1e-10:  # Избегаем деления на 0
        sinc_val = 1.0
    else:
        sinc_val = np.sin(arg) / arg
    
    # Экспоненциальный множитель
    exp_val = np.exp(1j * arg)
    
    return exp_val * sinc_val

def theoretical_sinr(epsilon, aD, N, ids, SNR_dB=20):
    P_signal = 1.0
    sigma_n2 = P_signal / (10**(SNR_dB/10))
    
    # Создаем массив той же длины, что и набор индексов
    sinr_per_subcarrier = np.zeros(len(ids))
    
    # Используем enumerate, чтобы k_val было реальным индексом (0, 10, 20...),
    # а idx — порядковым номером в массиве результатов (0, 1, 2...)
    for idx, k_val in enumerate(ids):
        # Полезная составляющая: используем k_val
        I_kk = calculate_I(k_val, k_val, epsilon, aD)
        useful_power = P_signal * np.abs(I_kk)**2
        
        interference_power = 0
        for n in range(N):
            if n != k_val:
                I_nk = calculate_I(n, k_val, epsilon, aD)
                interference_power += P_signal * np.abs(I_nk)**2
        
        sinr_linear = useful_power / (interference_power + sigma_n2)
        sinr_per_subcarrier[idx] = 10 * np.log10(sinr_linear)
        
    return sinr_per_subcarrier

# Пример использования
numer = 0
df = 15e3 * (2 ** numer)

# 1. Исходные данные
N = 2048
SNR_dB = 20
df = 15e3
fc = 1.6e9  - N * df / 2

ids = range(0, N, 50)

# Опорный расчет для dBFM (при V = 0)
ref_sinr_db = theoretical_sinr(0, 0, N, ids, SNR_dB)

# --- ПЕРВЫЙ ГРАФИК: SINR(k) в dBFM ---
vel = np.array([100, 1000, 4000, 8000])
epsilons = vel / 3e8
aD_vals = fc * epsilons / df

plt.figure(figsize=(12, 7))
markers = ['^', 's', 'o', 'd']

isEspilon = True
isDopler = False
for i in range(len(vel)):
    eps = epsilons[i] if isEspilon else 0
    ad = aD_vals[i] if isDopler else 0
    
    current_sinr = theoretical_sinr(eps, ad, N, ids, SNR_dB)
    # Перевод в dBFM: вычитаем опорный уровень в дБ
    dbfm_curve = current_sinr - ref_sinr_db
    
    plt.plot(ids, dbfm_curve, label=f'V = {vel[i]} м/с', 
             marker=markers[i], markerfacecolor='white', linewidth=2)

plt.xlabel('Индекс поднесущей (k)', fontsize=18)
plt.ylabel('Относительный SINR (dBFM), дБ', fontsize=18)
plt.title('Потери SINR относительно идеального случая (V=0)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('graph1_dbfm.png')

# --- ВТОРОЙ ГРАФИК: Средний SINR(V) в dBFM ---
v_range = np.arange(0, 8001, 50)
mean_dbfm = []

# Опорное среднее значение (линейное)
ref_mean_lin = np.mean(10**(ref_sinr_db / 10))

isEspilon = False
isDopler = True
for v in v_range:
    eps = (v / 3e8) if isEspilon else 0
    ad = (fc * (v / 3e8) / df) if isDopler else 0
    
    cur_sinr_db = theoretical_sinr(eps, ad, N, ids, SNR_dB)
    cur_mean_lin = np.mean(10**(cur_sinr_db / 10))
    
    # dBFM как отношение средних мощностей
    loss_db = 10 * np.log10(cur_mean_lin / ref_mean_lin)
    mean_dbfm.append(loss_db)
    print(f"Прогресс: {v}/8000 м/с", end='\r')

plt.figure(figsize=(12, 7))
plt.plot(v_range, mean_dbfm, 'r-o', linewidth=2, label='Средние потери по спектру')
plt.axhline(y=-3, color='black', linestyle='--', label='Порог -3 дБ')

plt.xlabel('Скорость (V), м/с', fontsize=18)
plt.ylabel('Средний dBFM, дБ', fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('graph2_dbfm.png')
plt.show()

