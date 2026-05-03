import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# --- ФУНКЦИИ РАСЧЕТА ---

def fast_theoretical_sinr(epsilon, aD, N, ids, SNR_dB=20):
    """Векторизованный расчет SINR для всех выбранных поднесущих сразу"""
    P_signal = 1.0
    sigma_n2 = P_signal / (10**(SNR_dB/10))
    
    # Сетки индексов: n (строки, все поднесущие), k (столбцы, выбранные ids)
    n = np.arange(N).reshape(-1, 1)
    k = np.array(ids).reshape(1, -1)
    
    # Аргумент для sinc и экспоненты
    arg = np.pi * ((1 + epsilon) * n + aD - k)
    
    # Матрица коэффициентов I[n, k]. np.sinc(x) = sin(pi*x)/(pi*x)
    I_matrix = np.exp(1j * arg) * np.sinc(arg / np.pi)
    powers = np.abs(I_matrix)**2
    
    # Полезная мощность: диагональные элементы (где n соответствует k)
    # Так как k — это подмножество n, извлекаем нужные элементы
    useful_power = np.array([powers[k_val, i] for i, k_val in enumerate(ids)])
    
    # Интерференция: сумма по столбцу (все n) минус полезный сигнал
    total_power_per_k = np.sum(powers, axis=0)
    interference_power = total_power_per_k - useful_power
    
    sinr_linear = useful_power / (interference_power + sigma_n2)
    return 10 * np.log10(sinr_linear)

def compute_step(v, N, ids, SNR_dB, fc, df, ref_sinr_db):
    """Функция для одного процесса: расчет средней потери при скорости v"""
    eps = v / 3e8
    ad = fc * eps / df
    ad = 0
    current_sinr = fast_theoretical_sinr(eps, ad, N, ids, SNR_dB)
    return np.mean(current_sinr - ref_sinr_db)

# --- ОСНОВНОЙ БЛОК ---

if __name__ == '__main__':
    # Параметры системы
    N = 2048
    SNR_dB = 20
    df = 15e3
    fc = 1.6e9 - N * df / 2
    ids = np.arange(0, N, 50)
    
    # Опорный расчет (V=0)
    ref_sinr_db = fast_theoretical_sinr(0, 0, N, ids, SNR_dB)
    
    # Параметры оси скоростей (высокое разрешение)
    vel_fine = np.linspace(0, 8000, 100) 
    
    print(f"Запуск параллельного расчета для {len(vel_fine)} точек...")
    
    # Параллельный расчет
    # max_workers можно не указывать, по умолчанию — количество ядер CPU
    with ProcessPoolExecutor() as executor:
        # Передаем фиксированные параметры через lambda или доп. аргументы
        from functools import partial
        worker_func = partial(compute_step, N=N, ids=ids, SNR_dB=SNR_dB, 
                             fc=fc, df=df, ref_sinr_db=ref_sinr_db)
        
        avg_dbfm_list = list(executor.map(worker_func, vel_fine))

    print("Расчет окончен. Сохранение данных и отрисовка...")

    # Сохранение для MATLAB
    data_to_save = {
        'vel_fine': vel_fine,
        'avg_dbfm': np.array(avg_dbfm_list)
    }
    scipy.io.savemat('graph_data_velocity.mat', data_to_save)

    # Отрисовка компактного графика
    plt.figure(figsize=(8, 5))
    plt.plot(vel_fine, avg_dbfm_list, color='black', linewidth=2)
    plt.fill_between(vel_fine, avg_dbfm_list, color='gray', alpha=0.1)
    
    plt.xlabel('Скорость $V$, м/с', fontsize=16)
    plt.ylabel('Средние потери $R$, дБ', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig('graph_velocity_parallel.png', dpi=300)
    plt.show()
