import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def compute_ici_matrix(N, a):
    """
    Вычисление матрицы интерференции ICI для OFDM с доплером
    
    Parameters:
    N - количество поднесущих
    a - коэффициент доплера (относительное изменение частоты)
    
    Returns:
    C - матрица интерференции размера N x N
    """
    C = np.zeros((N, N), dtype=complex)
    
    for k in range(N):
        for n in range(N):
            # Вычисление I[n,k] по формуле
            arg = np.pi * ((1 + a) * n - k)
            
            # sinc(πx) = sin(πx)/(πx)
            if np.abs(arg) < 1e-10:
                sinc_val = 1.0
            else:
                sinc_val = np.sin(arg) / arg
            
            # Экспоненциальный множитель
            exp_val = np.exp(1j * arg)
            
            C[k, n] = exp_val * sinc_val
    
    return C

def simulate_ofdm_system_matrix_method(N=64, mod_order=16, a_values=None, 
                                       SNR_dB_values=None, num_symbols=1000):
    """
    Быстрое моделирование OFDM системы с доплером через матричный метод
    """
    if a_values is None:
        a_values = np.logspace(-5, -2, 20)
    
    if SNR_dB_values is None:
        SNR_dB_values = [10, 15, 20, 25, 30]
    
    # Создание созвездия
    if mod_order == 16:
        # 16-QAM созвездие
        constellation = np.array([-3-3j, -3-1j, -3+1j, -3+3j,
                                  -1-3j, -1-1j, -1+1j, -1+3j,
                                  1-3j, 1-1j, 1+1j, 1+3j,
                                  3-3j, 3-1j, 3+1j, 3+3j])
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    elif mod_order == 4:
        # QPSK созвездие
        constellation = np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)
    elif mod_order == 64:
        # 64-QAM созвездие
        levels = [-7, -5, -3, -1, 1, 3, 5, 7]
        constellation = np.array([x + 1j*y for x in levels for y in levels])
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    
    # Для битовых ошибок нужно отображение символов на биты
    symbol_to_bits = {}
    bits_per_symbol = int(np.log2(mod_order))
    
    # Простое отображение (Gray coding можно реализовать при необходимости)
    for i in range(mod_order):
        # Преобразование в двоичное представление
        bits = np.array([int(b) for b in format(i, f'0{bits_per_symbol}b')])
        symbol_to_bits[i] = bits
    
    results = {
        'SINR': np.zeros((len(a_values), len(SNR_dB_values))),
        'BER': np.zeros((len(a_values), len(SNR_dB_values))),
        'SER': np.zeros((len(a_values), len(SNR_dB_values)))
    }
    
    for i, a in enumerate(a_values):
        print(f"Моделирование для a = {a:.6f} ({i+1}/{len(a_values)})")
        
        # Предварительно вычисляем матрицу интерференции
        C = compute_ici_matrix(N, a)
        diag_C = np.diag(C)  # Диагональные элементы для выравнивания
        
        for j, SNR_dB in enumerate(SNR_dB_values):
            SINR_sum = 0
            error_bits_sum = 0
            error_symbols_sum = 0
            total_bits_sum = 0
            total_symbols_sum = 0
            
            for sym_idx in range(num_symbols):
                # Генерация случайных символов
                symbol_indices = np.random.randint(0, mod_order, N)
                X = constellation[symbol_indices]
                
                # Применение интерференции
                Y_noiseless = C @ X
                
                # Добавление шума
                signal_power = np.mean(np.abs(Y_noiseless)**2)
                noise_power = signal_power / (10**(SNR_dB/10))
                noise = np.sqrt(noise_power/2) * (
                    np.random.randn(N) + 1j*np.random.randn(N))
                Y = Y_noiseless + noise
                
                # Оценка SINR
                useful_component = np.diag(C) * X
                interference_component = Y_noiseless - useful_component
                
                useful_power = np.mean(np.abs(useful_component)**2)
                interference_power = np.mean(np.abs(interference_component)**2)
                
                SINR = useful_power / (interference_power + noise_power)
                SINR_sum += 10*np.log10(SINR)
                
                # Приемник с простым выравниванием
                Y_eq = Y / diag_C
                
                # Демодуляция: поиск ближайшего символа в созвездии
                decisions = np.zeros(N, dtype=int)
                for idx in range(N):
                    # Расстояния до всех символов созвездия
                    distances = np.abs(Y_eq[idx] - constellation)**2
                    decisions[idx] = np.argmin(distances)
                
                # Расчет ошибок
                symbol_errors = np.sum(decisions != symbol_indices)
                error_symbols_sum += symbol_errors
                total_symbols_sum += N
                
                # Расчет битовых ошибок
                for sym_idx in range(N):
                    transmitted_bits = symbol_to_bits[symbol_indices[sym_idx]]
                    received_bits = symbol_to_bits[decisions[sym_idx]]
                    error_bits_sum += np.sum(transmitted_bits != received_bits)
                    total_bits_sum += bits_per_symbol
            
            # Усреднение результатов
            results['SINR'][i, j] = SINR_sum / num_symbols
            results['BER'][i, j] = error_bits_sum / total_bits_sum if total_bits_sum > 0 else 0
            results['SER'][i, j] = error_symbols_sum / total_symbols_sum if total_symbols_sum > 0 else 0
    
    return results, a_values, SNR_DB_values

def plot_ber_curves(results, a_values, SNR_DB_values):
    """
    Построение графиков BER в зависимости от a и SNR
    """
    # 1. BER vs SNR для разных значений a
    plt.figure(figsize=(14, 10))
    
    # Выберем несколько интересных значений a для отображения
    a_indices = [0, len(a_values)//4, len(a_values)//2, 3*len(a_values)//4, -1]
    selected_a = [a_values[i] for i in a_indices]
    
    for i, a_idx in enumerate(a_indices):
        plt.semilogy(SNR_DB_values, results['BER'][a_idx, :], 
                    marker='o', linewidth=2, markersize=8,
                    label=f'a = {a_values[a_idx]:.6f}')
    
    plt.xlabel('SNR, дБ', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title('Зависимость BER от SNR для разных коэффициентов доплера', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim([1e-6, 1])
    plt.tight_layout()
    plt.show()
    
    # 2. BER vs a для разных SNR
    plt.figure(figsize=(14, 10))
    
    for j, snr in enumerate(SNR_DB_values):
        plt.semilogy(a_values, results['BER'][:, j], 
                    marker='s', linewidth=2, markersize=6,
                    label=f'SNR = {snr} дБ')
    
    plt.xlabel('Коэффициент доплера (a)', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title('Зависимость BER от коэффициента доплера для разных SNR', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.xscale('log')
    plt.ylim([1e-6, 1])
    plt.tight_layout()
    plt.show()
    
    # 3. 3D график BER vs a vs SNR
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Создание сетки
    A_grid, SNR_grid = np.meshgrid(a_values, SNR_DB_values)
    BER_grid = results['BER'].T  # Транспонируем для согласования размерностей
    
    # Поверхностный график
    surf = ax.plot_surface(np.log10(A_grid), SNR_grid, np.log10(BER_grid + 1e-10),
                          cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('log10(a)', fontsize=12)
    ax.set_ylabel('SNR, дБ', fontsize=12)
    ax.set_zlabel('log10(BER)', fontsize=12)
    ax.set_title('3D зависимость BER от a и SNR', fontsize=14)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()
    
    # 4. График SINR vs a
    plt.figure(figsize=(12, 8))
    
    for j, snr in enumerate(SNR_DB_values):
        plt.plot(a_values, results['SINR'][:, j], 
                marker='^', linewidth=2, markersize=6,
                label=f'SNR = {snr} дБ')
    
    plt.xlabel('Коэффициент доплера (a)', fontsize=12)
    plt.ylabel('SINR, дБ', fontsize=12)
    plt.title('Зависимость SINR от коэффициента доплера', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Параметры моделирования
    N = 64
    mod_order = 16  # 16-QAM
    a_values = np.logspace(-5, -2, 15)  # 15 значений от 1e-5 до 1e-2
    SNR_DB_values = [10, 15, 20, 25, 30]
    num_symbols = 500  # Уменьшено для скорости, можно увеличить для точности
    
    print("Запуск моделирования OFDM системы с доплером...")
    print(f"Параметры: N={N}, Модуляция: {mod_order}-QAM")
    print(f"Коэффициенты доплера: от {a_values[0]:.2e} до {a_values[-1]:.2e}")
    print(f"Значения SNR: {SNR_DB_values} дБ")
    print(f"Количество символов: {num_symbols}")
    
    # Запуск моделирования
    results, a_values, SNR_DB_values = simulate_ofdm_system_matrix_method(
        N=N, mod_order=mod_order, a_values=a_values,
        SNR_dB_values=SNR_DB_values, num_symbols=num_symbols
    )
    
    # Построение графиков
    plot_ber_curves(results, a_values, SNR_DB_values)
    
    # Дополнительно: теоретическая BER для 16-QAM без доплера
    plt.figure(figsize=(10, 6))
    snr_linear = 10**(np.array(SNR_DB_values)/10)
    # Теоретическая аппроксимация BER для 16-QAM
    theoretical_ber = (3/8) * special.erfc(np.sqrt(snr_linear/5))
    plt.semilogy(SNR_DB_values, theoretical_ber, 'k--', linewidth=3, label='Теоретическая 16-QAM (без доплера)')
    plt.xlabel('SNR, дБ', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title('Теоретическая BER для 16-QAM без доплера', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()