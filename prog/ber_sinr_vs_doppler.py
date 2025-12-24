import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def compute_ici_matrix_fast(N, a):
    """
    Быстрое вычисление матрицы интерференции ICI для OFDM с доплером
    """
    n = np.arange(N).reshape(1, -1)
    k = np.arange(N).reshape(-1, 1)
    
    arg = np.pi * ((1 + a) * n - k)
    
    # Векторизованное вычисление sinc
    sinc_val = np.ones_like(arg, dtype=complex)
    mask = np.abs(arg) > 1e-10
    sinc_val[mask] = np.sin(arg[mask]) / arg[mask]
    
    # Экспоненциальный множитель
    exp_val = np.exp(1j * arg)
    
    return exp_val * sinc_val

def calculate_ber_single_doppler(N=8, mod_order=16, a=0, 
                                 SNR_dB_values=None, num_symbols=1000):
    """
    Расчет BER для одного значения доплера
    """
    if SNR_dB_values is None:
        SNR_dB_values = list(range(-10, 11, 1))
    
    # Создание созвездия 16-QAM
    if mod_order == 16:
        constellation = np.array([-3-3j, -3-1j, -3+1j, -3+3j,
                                  -1-3j, -1-1j, -1+1j, -1+3j,
                                  1-3j, 1-1j, 1+1j, 1+3j,
                                  3-3j, 3-1j, 3+1j, 3+3j])
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    else:
        raise ValueError(f"Неподдерживаемый порядок модуляции: {mod_order}")
    
    # Подготовка для расчета BER
    bits_per_symbol = int(np.log2(mod_order))
    
    # Предварительно вычисляем матрицу интерференции
    C = compute_ici_matrix_fast(N, a)
    diag_C = np.diag(C)
    
    # Массивы для результатов
    ber_values = np.zeros(len(SNR_dB_values))
    
    # Моделирование для каждого SNR
    for j, snr_dB in enumerate(SNR_dB_values):
        print(f"{j}/{len(SNR_dB_values)}")
        error_bits = 0
        total_bits = 0
        
        snr_lin = 10 ** (snr_dB / 10)
        for _ in range(num_symbols):
            
            # Генерация случайных символов
            symbol_indices = np.random.randint(0, mod_order, N)
            X = constellation[symbol_indices]
            
            # Применение интерференции
            Y_noiseless = C @ X
            
            # Добавление шума
            signal_energy = np.sum(np.abs(Y_noiseless)**2)
            noise_power = 1 / (np.sqrt(2))
            noise = np.sqrt(noise_power/2) * (
                np.random.randn(N) + 1j*np.random.randn(N))
            
            noise_energy = np.sum(np.abs(noise)**2)

            alfa = np.sqrt(signal_energy / noise_energy / snr_lin )

            Y = Y_noiseless + alfa * noise
            
            # Простой приемник
            Y_eq = Y #/ diag_C
            
            # Демодуляция
            decisions = np.zeros(N, dtype=int)
            for idx in range(N):
                distances = np.abs(Y_eq[idx] - constellation)**2
                decisions[idx] = np.argmin(distances)
            
            # Расчет битовых ошибок
            for sym_idx in range(N):
                transmitted_symbol = symbol_indices[sym_idx]
                received_symbol = decisions[sym_idx]
                
                # Подсчет различающихся битов
                transmitted_bits = format(transmitted_symbol, f'0{bits_per_symbol}b')
                received_bits = format(received_symbol, f'0{bits_per_symbol}b')
                
                for bit_idx in range(bits_per_symbol):
                    if transmitted_bits[bit_idx] != received_bits[bit_idx]:
                        error_bits += 1
                
                total_bits += bits_per_symbol
        
        # Расчет BER для данного SNR
        ber_values[j] = error_bits / total_bits if total_bits > 0 else 0
    
    return ber_values, SNR_dB_values

def theoretical_ber_16qam(SNR_dB_values):
    """
    Теоретическая BER для 16-QAM
    """
    snr_linear = 10**(np.array(SNR_dB_values) / 10)
    
    # Правильная формула для 16-QAM с Gray coding
    # BER = (3/4) * Q(√(SNR/10)), где Q(x) = 0.5*erfc(x/√2)
    ber = (3/4) * 0.5 * special.erfc(np.sqrt(snr_linear/10))#(3/4) * 0.5* special.erfc(np.sqrt(snr_linear/10)/ np.sqrt(2))
    
    print("ber theor")
    print(ber)
    return ber

def plot_ber_comparison(ber_simulated, ber_theoretical, SNR_dB_values, a):
    """
    Построение графика сравнения смоделированной и теоретической BER
    """
    plt.figure(figsize=(12, 8))
    
    # Смоделированная BER
    plt.semilogy(SNR_dB_values, ber_simulated, 
                'b-o', linewidth=2, markersize=8, markerfacecolor='white',
                label=f'Смоделированная (a={a})')
    
    # Теоретическая BER
    plt.semilogy(SNR_dB_values, ber_theoretical, 
                'r--', linewidth=3, 
                label='Теоретическая 16-QAM (без доплера)')
    
    # Настройка графика
    plt.xlabel('SNR, дБ', fontsize=14)
    plt.ylabel('BER', fontsize=14)
    plt.title(f'Сравнение BER для OFDM системы\nN=8, 16-QAM, a={a}', fontsize=16)
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    plt.ylim([1e-4, 1])
    
    # Добавление сетки
    ax = plt.gca()
    ax.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    plt.show()
    filename = f'ber_ofdm_100.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

# Основная функция
if __name__ == "__main__":
    # Параметры
    N = 1024
    mod_order = 16
    
    # Одно значение доплера (v = 0 м/c)
    vel = 0  # м/c
    c = 3e8  # скорость света, м/с
    a = vel / c
    
    # Диапазон SNR
    minSnr = -10
    maxSnr = 0
    snrStep = 1
    SNR_dB_values = list(range(minSnr, maxSnr + snrStep, snrStep))
    
    # Количество символов для усреднения
    num_symbols = 250
    
    print("=" * 60)
    print("Моделирование OFDM системы")
    print("=" * 60)
    print(f"Количество поднесущих: N = {N}")
    print(f"Модуляция: {mod_order}-QAM")
    print(f"Скорость: v = {vel} м/с")
    print(f"Коэффициент доплера: a = v/c = {a:.2e}")
    print(f"Диапазон SNR: от {minSnr} до {maxSnr} дБ с шагом {snrStep}")
    print(f"Количество символов: {num_symbols}")
    print("=" * 60)
    
    # Расчет смоделированной BER
    print("\nРасчет смоделированной BER...")
    ber_simulated, SNR_dB_values = calculate_ber_single_doppler(
        N=N, mod_order=mod_order, a=a,
        SNR_dB_values=SNR_dB_values, num_symbols=num_symbols
    )
    
    print("ber_sim")
    print(ber_simulated)

    print("SNR_dB_values")
    print(SNR_dB_values)

    # Расчет теоретической BER
    print("Расчет теоретической BER...")
    ber_theoretical = theoretical_ber_16qam(SNR_dB_values)
    
    # Построение графика сравнения
    print("Построение графика...")
    plot_ber_comparison(ber_simulated, ber_theoretical, SNR_dB_values, a)
    
    # Вывод таблицы результатов
    #print("\n" + "=" * 60)
    #print("Результаты моделирования:")
    #print("=" * 60)
    #print(f"{'SNR (дБ)':<10} {'BER (модель)':<15} {'BER (теор.)':<15} {'Отношение':<12}")
    #print("-" * 60)
    #
    #for i, snr in enumerate(SNR_dB_values):
    #    ratio = ber_simulated[i] / ber_theoretical[i] if ber_theoretical[i] > 0 else np.nan
    #    print(f"{snr:<10} {ber_simulated[i]:<15.2e} {ber_theoretical[i]:<15.2e} {ratio:<12.2f}")
    #
    ## Расчет среднего отклонения
    #valid_indices = ber_theoretical > 0
    #if np.any(valid_indices):
    #    avg_ratio = np.mean(ber_simulated[valid_indices] / ber_theoretical[valid_indices])
    #    print("-" * 60)
    #    print(f"Среднее отношение (модель/теория): {avg_ratio:.2f}")