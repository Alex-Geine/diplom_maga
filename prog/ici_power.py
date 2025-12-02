import numpy as np
import matplotlib.pyplot as plt

def compute_ici_matrix(N, a):
    """
    Вычисление матрицы интерференции с точной формулой
    
    Parameters:
    N - число поднесущих
    a - коэффициент доплеровского масштабирования
    """
    C = np.zeros((N, N), dtype=complex)
    
    # Для симметрии центрируем частоты вокруг 0
    # Это удобно для визуализации
    n_indices = np.arange(N) - N//2  # [-N/2, ..., N/2-1]
    k_indices = np.arange(N) - N//2
    
    for i, k in enumerate(k_indices):      # приемная поднесущая
        for j, n in enumerate(n_indices):  # передаваемая поднесущая
            delta = (1 + a) * n - k
            if delta == 0:
                C[i, j] = 1.0
            else:
                C[i, j] = np.exp(1j * np.pi * delta) * np.sinc(delta)
    
    return C

def analyze_ici_power_distribution(N=64, a=0.01):
    """
    Анализ распределения мощности ICI по поднесущим
    """
    C = compute_ici_matrix(N, a)
    
    # Мощность полезного сигнала для каждой поднесущей
    useful_power = np.abs(np.diag(C))**2
    
    # Мощность ICI для каждой поднесущей
    ici_power_per_subcarrier = np.sum(np.abs(C - np.diag(np.diag(C)))**2, axis=1)
    
    # Относительная мощность ICI
    ici_relative = ici_power_per_subcarrier / useful_power
    
    plt.figure(figsize=(12, 8))
    
    # График 1: Амплитуда диагональных элементов
    plt.subplot(2, 2, 1)
    plt.plot(np.abs(np.diag(C)), 'b-', linewidth=2)
    plt.xlabel('Индекс поднесущей')
    plt.ylabel('|C[k,k]|')
    plt.title('Ослабление полезного сигнала')
    plt.grid(True)
    
    # График 2: Фаза диагональных элементов
    plt.subplot(2, 2, 2)
    plt.plot(np.angle(np.diag(C)), 'r-', linewidth=2)
    plt.xlabel('Индекс поднесущей')
    plt.ylabel('Фаза C[k,k] (рад)')
    plt.title('Фазовый наклон из-за доплера')
    plt.grid(True)
    
    # График 3: Мощность ICI по поднесущим
    plt.subplot(2, 2, 3)
    plt.plot(10*np.log10(ici_power_per_subcarrier + 1e-12), 'g-', linewidth=2)
    plt.xlabel('Индекс поднесущей')
    plt.ylabel('Мощность ICI, дБ')
    plt.title('Распределение мощности ICI')
    plt.grid(True)
    
    # График 4: Срез матрицы для одной поднесущей
    plt.subplot(2, 2, 4)
    k_center = N//2
    plt.plot(np.abs(C[k_center, :]), 'm-', linewidth=2)
    plt.axvline(x=k_center, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Передаваемая поднесущая n')
    plt.ylabel('|C[k_center, n]|')
    plt.title(f'Влияние всех поднесущих на поднесущую {k_center}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Общая статистика
    total_ici_power = np.sum(ici_power_per_subcarrier)
    total_useful_power = np.sum(useful_power)
    ici_ratio_db = 10*np.log10(total_ici_power / total_useful_power)
    
    print(f"Общая мощность ICI: {total_ici_power:.2e}")
    print(f"Общая мощность полезного сигнала: {total_useful_power:.2e}")
    print(f"Отношение ICI/сигнал: {ici_ratio_db:.2f} дБ")

a = 8000/3e8  # наш коэффициент
analyze_ici_power_distribution(64, a)