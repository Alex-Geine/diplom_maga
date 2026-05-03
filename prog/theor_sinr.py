import numpy as np
import scipy.io
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

isOld = True
isNew = False

ids = range(0, N, 50)

# Опорный расчет для dBFM (при V = 0)
ref_sinr_db = theoretical_sinr(0, 0, N, ids, SNR_dB)

# --- ПЕРВЫЙ ГРАФИК: SINR(k) в dBFM ---
vel = np.array([100, 1000, 4000, 8000])
epsilons = vel / 3e8
aD_vals = fc * epsilons / df

plt.figure(figsize=(12, 7))
markers = ['^', 's', 'o', 'd']

data_to_save = {
        'subcarriers': ids,
        'velocities': vel,
        'curves': []
        }

if isOld:
    # isEspilon = True
    # isDopler = False
    # curv_list = []
    # for i in range(len(vel)):
    #     eps = epsilons[i] if isEspilon else 0
    #     ad = aD_vals[i] if isDopler else 0



    #     current_sinr = theoretical_sinr(eps, ad, N, ids, SNR_dB)
    #     # Перевод в dBFM: вычитаем опорный уровень в дБ
    #     dbfm_curve = current_sinr - ref_sinr_db
    #     curv_list.append(dbfm_curve)

    #     plt.plot(ids, dbfm_curve, label=f'V = {vel[i]} м/с', 
    #              marker=markers[i], markerfacecolor='white', linewidth=2, color='black', markersize=8)

    # data_to_save['curves'] = curv_list

    # plt.xlabel('Индекс поднесущей (k)', fontsize=24)
    # plt.tick_params(axis='both', labelsize=14)
    # plt.ylabel('R, дБ', fontsize=24)
    # # plt.title('Потери SINR относительно идеального случая (V=0)', fontsize=16)
    # plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=14)
    # plt.savefig('graph1_dbfm.png')
    # scipy.io.savemat('graph_data.mat', data_to_save)

    # --- ВТОРОЙ ГРАФИК: Average SINR vs Velocity ---
    # Зададим более плотный диапазон скоростей для плавного графика
    vel_fine = np.linspace(0, 8000, 20) 
    eps_fine = vel_fine / 3e8
    ad_fine = fc * eps_fine / df

    avg_dbfm_list = []

    for i in range(len(vel_fine)):
        # Вычисляем SINR для всех поднесущих из ids
        current_sinr = theoretical_sinr(eps_fine[i], ad_fine[i], N, ids, SNR_dB)
        # Вычисляем потери (dBFM) и усредняем
        avg_dbfm = np.mean(current_sinr - ref_sinr_db)
        avg_dbfm_list.append(avg_dbfm)

    # Добавляем в структуру для сохранения
    data_to_save['vel_fine'] = vel_fine
    data_to_save['avg_dbfm'] = avg_dbfm_list

    # Отрисовка второго графика
    plt.figure(figsize=(10, 6))
    plt.plot(vel_fine, avg_dbfm_list, 'k-o', linewidth=2, markerfacecolor='white')
    plt.xlabel('Скорость V, м/с', fontsize=18)
    plt.ylabel('Средние потери R, дБ', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph2_velocity.png')

    # Пересохраняем .mat файл с новыми полями
    scipy.io.savemat('graph_data.mat', data_to_save)

# --- ТРЕТИЙ ГРАФИК: Влияние fc и df на SINR(V) ---
v_range_3 = np.arange(0, 8001, 1) # Шаг 200 для ускорения расчета
scenarios = [
    {'fc': 1.6e9,  'df': 15e3, 'style' : '-', 'color' : 'black', 'label': 'fc = 1.6 ГГц, Δf = 15 кГц'},
    {'fc': 1.6e9,  'df': 240e3, 'style' : '--', 'color' : 'black', 'label': 'fc = 1.6 ГГц, Δf = 240 кГц'},
    {'fc': 20e9,   'df': 15e3,  'style' : '-', 'color' : 'grey', 'label': 'fc = 20 ГГц, Δf = 15 кГц'},
    {'fc': 20e9,   'df': 240e3, 'style' : '--', 'color' : 'grey', 'label': 'fc = 20 ГГц, Δf = 240 кГц'},
]

plt.figure(figsize=(12, 7))
# styles = ['-', '--']
# colors = ['black', 'grey']

# markers = ['^', 's', 'o', 'd']

if isNew:
    isEspilon = False
    isDopler = True
    for idx, sc in enumerate(scenarios):
        df_curr = sc['df']
        f_curr = sc['fc'] - N * df_curr / 2

        # Расчет опорного значения (V=0) для данного сценария
        ref_sinr_db_curr = theoretical_sinr(0, 0, N, ids, SNR_dB)
        ref_mean_lin_curr = np.mean(10**(ref_sinr_db_curr / 10))

        losses = []
        for v in v_range_3:
            # Учитываем оба фактора (эпсилон и Доплер) для реалистичности
            eps = v / 3e8
            ad = (f_curr * eps) / df_curr

            cur_sinr_db = theoretical_sinr(eps, ad, N, ids, SNR_dB)
            cur_mean_lin = np.mean(10**(cur_sinr_db / 10))

            loss_db = 10 * np.log10(cur_mean_lin / ref_mean_lin_curr)
            losses.append(loss_db)
            print(f"Сценарий {idx+1}/4 | Скорость: {v}/8000 м/с    ", end='\r')

        plt.plot(v_range_3, losses, label=sc['label'], 
                 linewidth=2.5, linestyle=sc['style'], color=sc['color'])#, marker=markers[idx], markerfacecolor='white', markersize=6)

    plt.axhline(y=-3, color='red', linestyle='--', alpha=0.5, label='SINR -3 дБ')
    plt.xlabel('Относительная скорость, м/с', fontsize=24)
    plt.ylabel('SINR, дБ', fontsize=24)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16, loc='upper right')
    plt.savefig('graph3_comparative.png')
    plt.show()