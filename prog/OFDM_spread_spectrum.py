import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

def sinc_function(x, center=0, width=1):
    """Функция sinc для представления поднесущих"""
    return np.sinc((x - center) / width)

def plot_ofdm_spectrum(fd, v, n_subcarriers, c=3e8):
    """
    Построение спектра OFDM сигнала с эффектом Доплера
    
    Parameters:
    fd - расстояние между поднесущими (Гц)
    v - радиальная скорость (м/с)
    n_subcarriers - количество поднесущих
    c - скорость света (м/с)
    """
    
    # Создаем частотную ось
    f_min = -fd * 2
    f_max = fd * (n_subcarriers + 2)
    f = np.linspace(f_min, f_max, 10000)
    
    # Доплеровский множитель
    doppler_factor = 1 + v/c
    
    plt.figure(figsize=(12, 8))
    
    # Исходный спектр (без доплера)
    original_spectrum = np.zeros_like(f)
    for n in range(1, n_subcarriers + 1):
        center_freq = fd * n
        original_spectrum += sinc_function(f, center_freq, fd/2)
    
    plt.plot(f, original_spectrum, 'b-', linewidth=2, label='Исходный спектр')
    
    # Спектр с доплеровским смещением
    doppler_spectrum = np.zeros_like(f)
    for n in range(1, n_subcarriers + 1):
        center_freq = fd * n * doppler_factor  # Смещенная частота
        doppler_spectrum += sinc_function(f, center_freq, fd/2)
    
    plt.plot(f, doppler_spectrum, 'r--', linewidth=2, label=f'С доплером (v={v} м/с)')
    
    # Отмечаем положения поднесущих
    original_centers = [fd * n for n in range(1, n_subcarriers + 1)]
    doppler_centers = [fd * n * doppler_factor for n in range(1, n_subcarriers + 1)]
    
    for center in original_centers:
        plt.axvline(x=center, color='blue', linestyle=':', alpha=0.5)
    
    for center in doppler_centers:
        plt.axvline(x=center, color='red', linestyle=':', alpha=0.5)
    
    # Настройки графика
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title(f'Доплеровское уширение OFDM спектра\nfd={fd} Гц, v={v} м/с, поднесущих={n_subcarriers}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(f_min, f_max)
    
    # Добавляем информацию о смещении
    shift = (doppler_centers[0] - original_centers[0]) if n_subcarriers > 0 else 0
    plt.text(0.02, 0.98, f'Смещение: {shift:.2f} Гц\nДоплер множитель: {doppler_factor:.6f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def interactive_plot():
    """Интерактивный режим с вводом параметров"""
    print("Демонстрация доплеровского уширения OFDM спектра")
    print("=" * 50)
    
    try:
        fd = float(input("Введите расстояние между поднесущими (fd в Гц): "))
        v = float(input("Введите радиальную скорость (v в м/с): "))
        n_subcarriers = int(input("Введите количество поднесущих (n): "))
        
        if n_subcarriers <= 0:
            print("Количество поднесущих должно быть положительным!")
            return
        
        plot_ofdm_spectrum(fd, v, n_subcarriers)
        
    except ValueError:
        print("Ошибка ввода! Пожалуйста, вводите числовые значения.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования с заданными параметрами
if __name__ == "__main__":
    # Параметры по умолчанию
    FD = 1000  # расстояние между поднесущими 1 кГц
    V = 10000  # радиальная скорость 10 км/с
    N_SUBCARRIERS = 8  # количество поднесущих
    
    # Запуск интерактивного режима
    interactive_choice = input("Запустить интерактивный режим? (y/n): ").lower()
    
    if interactive_choice == 'y':
        interactive_plot()
    else:
        # Использование параметров по умолчанию
        print(f"Используются параметры по умолчанию:")
        print(f"fd = {FD} Гц, v = {V} м/с, n = {N_SUBCARRIERS}")
        plot_ofdm_spectrum(FD, V, N_SUBCARRIERS)
    
    # Дополнительные примеры для демонстрации
    print("\nДополнительные примеры:")
    print("1. Медленное движение (v = 100 м/с):")
    plot_ofdm_spectrum(1000, 100, 5)
    
    print("2. Быстрое движение (v = 100000 м/с):")
    plot_ofdm_spectrum(1000, 100000, 5)
    
    print("3. Много поднесущих:")
    plot_ofdm_spectrum(500, 5000, 12)