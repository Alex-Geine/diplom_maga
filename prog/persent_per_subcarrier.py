import numpy as np
import matplotlib.pyplot as plt

def plot_doppler_deviation_relative_to_fd(fd, v, n_subcarriers, c=3e8):
    """
    График отклонения частоты относительно fd от номера поднесущей
    
    Parameters:
    fd - расстояние между поднесущими (Гц)
    v - радиальная скорость (м/с)
    n_subcarriers - количество поднесущих
    c - скорость света (м/с)
    """
    
    # Доплеровский множитель
    doppler_factor = 1 + v/c
    
    # Номера поднесущих
    n_values = np.arange(1, n_subcarriers + 1)
    
    # Отклонение относительно fd для каждой поднесущей
    # Δf = f_doppler - f_original = fd * n * (v/c)
    # Относительное отклонение: Δf / fd = n * (v/c)
    deviation_relative = n_values * (v / c)
    deviation_percent = deviation_relative * 100
    
    plt.figure(figsize=(14, 8))
    
    # Основной график
    plt.plot(n_values, deviation_percent, 'bo-', linewidth=2, markersize=4, 
             label=f'Отклонение относительно fd')
    
    # Линейная аппроксимация
    slope = (v / c) * 100
    plt.plot(n_values, slope * n_values, 'r--', alpha=0.7, 
             label=f'Линейная зависимость: {slope:.6f}% × n')
    
    # Настройки графика
    plt.xlabel('Номер поднесущей (n)')
    plt.ylabel('Отклонение частоты относительно fd (%)')
    plt.title(f'Доплеровское отклонение частоты относительно fd\n'
              f'fd = {fd/1000:.1f} кГц, v = {v/1000:.1f} км/с, n_subcarriers = {n_subcarriers}')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Добавляем аннотации для некоторых точек
    annotation_points = [1, n_subcarriers//4, n_subcarriers//2, 3*n_subcarriers//4, n_subcarriers]
    for n in annotation_points:
        if n <= n_subcarriers:
            plt.annotate(f'n={n}: {deviation_percent[n-1]:.4f}%', 
                        xy=(n, deviation_percent[n-1]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8,
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Информационная панель
    info_text = (f'Параметры:\n'
                f'fd = {fd} Гц ({fd/1000:.1f} кГц)\n'
                f'v = {v} м/с ({v/1000:.1f} км/с)\n'
                f'n_subcarriers = {n_subcarriers}\n'
                f'c = {c:.0f} м/с\n'
                f'v/c = {v/c:.8f}\n'
                f'Наклон: {slope:.6f}% на номер поднесущей\n'
                f'Отклонение для n=1: {deviation_percent[0]:.6f}%\n'
                f'Отклонение для n={n_subcarriers}: {deviation_percent[-1]:.6f}%')
    
    plt.text(0.65, 0.25, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод таблицы данных
    print(f"\nТаблица отклонений относительно fd (fd={fd} Гц, v={v} м/с):")
    print("n\tΔf/fd\t\tОтн.отклонение(%)\tАбс.смещение(Гц)")
    print("-\t-----\t\t----------------\t----------------")
    
    for n in [1, 2, 3, n_subcarriers//2, n_subcarriers-1, n_subcarriers]:
        if n <= n_subcarriers:
            rel_deviation = n * (v / c)
            abs_shift = fd * n * (v / c)
            print(f"{n}\t{rel_deviation:.8f}\t{rel_deviation*100:.8f}%\t\t{abs_shift:.4f}")

def analyze_multiple_cases():
    """Анализ нескольких случаев с разными параметрами"""
    cases = [
        (15000, 8000, 1024, "Основной случай: fd=15кГц, v=8км/с"),
        (15000, 3000, 1024, "Меньшая скорость: v=3км/с"),
        (15000, 15000, 1024, "Большая скорость: v=15км/с"),
        (10000, 8000, 1024, "Меньшее fd: 10кГц"),
        (20000, 8000, 1024, "Большее fd: 20кГц"),
        (15000, 8000, 512, "Меньше поднесущих: 512"),
        (15000, 8000, 2048, "Больше поднесущих: 2048")
    ]
    
    for fd, v, n, description in cases:
        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")
        plot_doppler_deviation_relative_to_fd(fd, v, n)

def compare_different_speeds():
    """Сравнение разных скоростей для fd=15000, n=1024"""
    fd = 15000
    n_subcarriers = 1024
    speeds = [1000, 3000, 8000, 15000, 30000]  # м/с
    
    plt.figure(figsize=(12, 8))
    
    n_values = np.arange(1, n_subcarriers + 1)
    
    for v in speeds:
        deviation_percent = n_values * (v / 3e8) * 100
        plt.plot(n_values, deviation_percent, linewidth=2, 
                label=f'v = {v/1000:.1f} км/с (наклон: {v/3e8*100:.6f}%)')
    
    plt.xlabel('Номер поднесущей (n)')
    plt.ylabel('Отклонение частоты относительно fd (%)')
    plt.title(f'Сравнение отклонений для разных скоростей\nfd = {fd/1000:.1f} кГц, n = {n_subcarriers}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # Логарифмическая шкала для наглядности
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    print("Анализ доплеровского отклонения частоты относительно fd")
    print("Формула: Δf/fd = n × (v/c)")
    print("Отклонение линейно зависит от номера поднесущей")
    print("=" * 80)
    
    # Основной случай: fd=15000 Гц, v=8000 м/с, n=1024
    FD = 15000
    V = 8000
    N_SUBCARRIERS = 1024
    
    plot_doppler_deviation_relative_to_fd(FD, V, N_SUBCARRIERS)
    
    # Дополнительные случаи для сравнения
    analyze_multiple_cases()
    
    # Сравнение разных скоростей
    compare_different_speeds()
    
    # Физический анализ
    print(f"\n{'='*80}")
    print("ФИЗИЧЕСКИЙ АНАЛИЗ:")
    print(f"{'='*80}")
    print("Для формулы f_i = fd × n × (1 + v/c):")
    print("Δf = f_doppler - f_original = fd × n × (v/c)")
    print("Относительное отклонение относительно fd: Δf/fd = n × (v/c)")
    print("Таким образом, отклонение ЛИНЕЙНО растет с номером поднесущей!")
    print("Это означает, что высокочастотные поднесущие смещаются сильнее.")