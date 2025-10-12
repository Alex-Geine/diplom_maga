import numpy as np
import matplotlib.pyplot as plt

def plot_ofdm_doppler_spectrum(fd, v, n_subcarriers, c=3e8):
    """
    Построение спектра OFDM сигнала с эффектом Доплера
    Сдвинутые поднесущие: f_i = fd * n * (1 + v/c)
    
    Parameters:
    fd - расстояние между поднесущими (Гц)
    v - радиальная скорость (м/с)
    n_subcarriers - количество поднесущих
    c - скорость света (м/с)
    """
    
    # Создаем частотную ось
    f_min = 0
    f_max = fd * (n_subcarriers + 2)
    f = np.linspace(f_min, f_max, 5000)
    
    # Доплеровский множитель
    doppler_factor = 1 + v/c
    
    plt.figure(figsize=(12, 8))
    
    # Исходный спектр (без доплера)
    original_spectrum = np.zeros_like(f)
    original_centers = []
    for n in range(1, n_subcarriers + 1):
        center_freq = fd * n
        original_centers.append(center_freq)
        # Функция sinc для представления поднесущей
        sinc_val = np.sinc((f - center_freq) / (fd/3))
        original_spectrum += sinc_val
    
    plt.plot(f, original_spectrum, 'b-', linewidth=2, label='Исходный спектр')
    
    # Спектр с доплеровским смещением (ВСЕ поднесущие смещены)
    doppler_spectrum = np.zeros_like(f)
    doppler_centers = []
    for n in range(1, n_subcarriers + 1):
        center_freq = fd * n * doppler_factor  # Правильная формула
        doppler_centers.append(center_freq)
        # Функция sinc для представления поднесущей
        sinc_val = np.sinc((f - center_freq) / (fd/3))
        doppler_spectrum += sinc_val
    
    plt.plot(f, doppler_spectrum, 'r--', linewidth=2, label=f'С доплером (v={v} м/с)')
    
    # Отмечаем положения поднесущих
    for i, center in enumerate(original_centers):
        plt.axvline(x=center, color='blue', linestyle=':', alpha=0.5)
        plt.text(center, 1.05, f'n={i+1}', ha='center', color='blue', fontsize=8)
    
    for i, center in enumerate(doppler_centers):
        plt.axvline(x=center, color='red', linestyle=':', alpha=0.5)
        plt.text(center, 1.15, f'n={i+1}', ha='center', color='red', fontsize=8)
    
    # Настройки графика
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title(f'Доплеровское уширение OFDM спектра\nfd={fd} Гц, v={v} м/с, поднесущих={n_subcarriers}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.2, 1.3)
    
    # Добавляем информацию о смещении
    if n_subcarriers > 0:
        shift = doppler_centers[-1] - original_centers[-1]
        relative_shift = shift / fd * 100
        plt.text(0.02, 0.98, f'Смещение последней поднесущей: {shift:.2f} Гц\n'
                             f'Относительное смещение: {relative_shift:.1f}%\n'
                             f'Доплер множитель: {doppler_factor:.8f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Выводим таблицу смещений
    print(f"\nТаблица смещений поднесущих (fd={fd} Гц, v={v} м/с):")
    print("n\tИсходная\tСмещенная\tСмещение\tОтн.смещ.")
    print("-\t--------\t---------\t--------\t---------")
    for i in range(n_subcarriers):
        original = original_centers[i]
        doppler = doppler_centers[i]
        shift = doppler - original
        relative = shift / fd * 100
        print(f"{i+1}\t{original:.1f}\t\t{doppler:.1f}\t\t{shift:.2f}\t\t{relative:.2f}%")

def analyze_doppler_impact(fd, v, n_subcarriers, c=3e8):
    """
    Анализ влияния доплеровского смещения на прием
    """
    doppler_factor = 1 + v/c
    
    print(f"\nАнализ влияния доплера:")
    print(f"Расстояние между поднесущими: {fd} Гц")
    print(f"Радиальная скорость: {v} м/с")
    print(f"Доплеровский множитель: {doppler_factor:.8f}")
    
    # Смещение для каждой поднесущей
    for n in range(1, n_subcarriers + 1):
        original_freq = fd * n
        doppler_freq = fd * n * doppler_factor
        shift = doppler_freq - original_freq
        relative_shift = shift / fd * 100
        
        print(f"Поднесущая {n}: смещение = {shift:.2f} Гц ({relative_shift:.2f}%)")
        
        # Проверка на интерференцию
        if abs(shift) > fd * 0.1:  # Если смещение > 10% от расстояния
            print(f"  ⚠️  Критическое смещение! Возможна интерференция")

# Основная программа
if __name__ == "__main__":
    # Параметры по умолчанию
    FD = 15000  # 15 кГц расстояние между поднесущими
    V = 3000    # 3 км/с радиальная скорость
    N_SUBCARRIERS = 8  # количество поднесущих
    
    print("Демонстрация доплеровского уширения OFDM спектра")
    print("=" * 50)
    print("Формула для смещенных поднесущих: f_i = fd * n * (1 + v/c)")
    print("=" * 50)
    
    # Запуск с параметрами по умолчанию
    plot_ofdm_doppler_spectrum(FD, V, N_SUBCARRIERS)
    analyze_doppler_impact(FD, V, N_SUBCARRIERS)
    
    # Дополнительные примеры
    examples = [
        #(15000, 1000, 1024, "Медленное движение"),
        (15000, 10000, 1024, "Быстрое движение"),
        #(10000, 5000, 1024, "Узкое расстояние"),
        #(20000, 3000, 1024, "Широкое расстояние")
    ]
    
    for fd, v, n, desc in examples:
        print(f"\n{desc}: fd={fd} Гц, v={v} м/с")
        plot_ofdm_doppler_spectrum(fd, v, n)
        analyze_doppler_impact(fd, v, n)