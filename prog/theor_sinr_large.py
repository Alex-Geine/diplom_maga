import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def fast_sinr_mean_lin(v, N, ids, SNR_dB, fc_base, df):
    """Считает среднее линейное значение SINR для заданной скорости"""
    eps = v / 3e8
    # Учитываем сдвиг частоты fc
    fc_actual = fc_base - N * df / 2
    ad = (fc_actual * eps) / df
    
    P_signal = 1.0
    sigma_n2 = P_signal / (10**(SNR_dB/10))
    
    n = np.arange(N).reshape(-1, 1)
    k = np.array(ids).reshape(1, -1)
    eps = 0
    # Векторизованный расчет матрицы I
    arg = np.pi * ((1 + eps) * n + ad - k)
    I_matrix = np.exp(1j * arg) * np.sinc(arg / np.pi)
    powers = np.abs(I_matrix)**2
    
    useful_power = np.array([powers[k_val, i] for i, k_val in enumerate(ids)])
    total_power_per_k = np.sum(powers, axis=0)
    interference_power = total_power_per_k - useful_power
    
    sinr_linear = useful_power / (interference_power + sigma_n2)
    return np.mean(sinr_linear)

if __name__ == '__main__':
    N = 2048
    SNR_dB = 20
    ids = np.arange(0, N, 50)
    v_range = np.arange(0, 8001, 50) # Шаг 50 дает отличную плавность
    
    scenarios = [
        {'fc': 1e1,  'df': 15e3,  'style': '-',  'color': 'black', 'label': 'fc = 1e1 Hz, df = 15 kHz'},
        {'fc': 1e2,  'df': 15e3,  'style': '-',  'color': 'black', 'label': 'fc = 1e2 Hz, df = 15 kHz'},
        {'fc': 1e3,  'df': 15e3,  'style': '-',  'color': 'black', 'label': 'fc = 1e3 Hz, df = 15 kHz'},
        {'fc': 1e4,  'df': 15e3,  'style': '-',  'color': 'black', 'label': 'fc = 1e4 Hz, df = 15 kHz'},
        {'fc': 5e6,  'df': 15e3, 'style': '--', 'color': 'black', 'label':  'fc = 5e6 Hz, df = 15 kHz'},
        {'fc': 1e6,  'df': 15e3,  'style': '-',  'color': 'grey',  'label': 'fc = 1e6 Hz, df = 15 kHz'},
        {'fc': 1e7,  'df': 15e3, 'style': '--', 'color': 'grey',  'label':  'fc = 1e7 Hz, df = 15 kHz'},
        {'fc': 1e8,  'df': 15e3, 'style': '--', 'color': 'grey',  'label':  'fc = 1e8 Hz, df = 15 kHz'},
        {'fc': 1e9,  'df': 15e3, 'style': '--', 'color': 'grey',  'label':  'fc = 1e9 Hz, df = 15 kHz'},
    ]

    results_to_save = {'v_range': v_range, 'scenarios': []}
    plt.figure(figsize=(10, 6))

    for idx, sc in enumerate(scenarios):
        print(f"Расчет сценария {idx+1}/{len(scenarios)}: {sc['label']}")
        
        # Опорное значение при V=0
        ref_linear = fast_sinr_mean_lin(0, N, ids, SNR_dB, sc['fc'], sc['df'])
        
        # Параллельный расчет по скоростям
        worker = partial(fast_sinr_mean_lin, N=N, ids=ids, SNR_dB=SNR_dB, fc_base=sc['fc'], df=sc['df'])
        with ProcessPoolExecutor() as executor:
            mean_sinr_linear_list = list(executor.map(worker, v_range))
        
        # Перевод в дБ (относительно V=0)
        losses_db = 10 * np.log10(np.array(mean_sinr_linear_list) / ref_linear)
        # losses_db = 10 * np.log10(np.array(mean_sinr_linear_list))
        
        # Сохранение данных
        results_to_save['scenarios'].append({
            'label': sc['label'],
            'losses': losses_db,
            'style': sc['style'],
            'color': sc['color']
        })
        print(losses_db)

        # Отрисовка
        plt.plot(v_range, losses_db, label=sc['label'], 
                 linestyle=sc['style'], color=sc['color'], linewidth=2)

    # plt.axhline(y=-3, color='red', linestyle=':', alpha=0.7, label='Порог -3 дБ')
    plt.xlabel('Скорость, м/с', fontsize=14)
    plt.ylabel('Потери SINR, дБ', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('graph3_optimized1.png', dpi=300)
    
    scipy.io.savemat('graph3_data.mat', results_to_save)
    print("Готово. Данные сохранены в 'graph3_data.mat'")
