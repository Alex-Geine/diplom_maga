import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
import os
import datetime

# ------------------------------ Классы ------------------------------
class OFDMTx:
    """OFDM передатчик с BPSK и IFFT"""
    def __init__(self, fft_size):
        self.fft_size = fft_size

    def map(self, bits):
        """BPSK: 0 -> -1, 1 -> +1"""
        return 2 * bits - 1  # (0->-1, 1->1)

    def ifft(self, symbols):
        """Обратное БПФ с сохранением энергии (ортогональное преобразование)"""
        return np.fft.ifft(symbols, norm='ortho')

    def transmit(self, bits):
        """Полный цикл передачи: биты -> символы -> IFFT -> временной сигнал"""
        symbols = self.map(bits)
        signal = self.ifft(symbols)
        return signal


class OFDMRx:
    """OFDM приёмник с FFT и демодуляцией BPSK"""
    def __init__(self, fft_size):
        self.fft_size = fft_size

    def fft(self, signal):
        """Прямое БПФ с сохранением энергии"""
        return np.fft.fft(signal, norm='ortho')

    def demap(self, symbols):
        """Жёсткая демодуляция BPSK: real > 0 -> 1, иначе 0"""
        return (np.real(symbols) > 0).astype(int)


class Channel:
    """Канал с AWGN (плоский)"""
    def __init__(self, snr_db):
        self.snr_db = snr_db
        # Мощность сигнала на входе канала предполагается равной 1 (после ортонормированного IFFT)
        self.snr_lin = 10 ** (snr_db / 10.0)
        self.N0 = 1.0 / self.snr_lin  # дисперсия комплексного шума

    def add_noise(self, signal):
        """Добавить комплексный белый гауссов шум с дисперсией N0 на комплексную выборку"""
        noise = np.sqrt(self.N0 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise


# ------------------------------ Вспомогательные функции ------------------------------
def ber_awgn_theoretical(snr_lin):
    """Теоретическая BER для BPSK в AWGN"""
    return 0.5 * erfc(np.sqrt(snr_lin))

def bler_theoretical(bits_per_block, snr_lin):
    """Теоретическая BLER для блока независимых битов (BPSK, AWGN)"""
    ber = ber_awgn_theoretical(snr_lin)
    return 1 - (1 - ber) ** bits_per_block

def simulate_snr(snr_db, fft_size, num_trials):
    """Симуляция для одного значения SNR"""
    tx = OFDMTx(fft_size)
    rx = OFDMRx(fft_size)
    channel = Channel(snr_db)

    total_bit_errors = 0
    total_blocks = 0
    block_errors = 0
    throughput_per_trial = []  # доля успешных битов в каждом опыте

    for _ in range(num_trials):
        # Генерация случайных бит
        bits_tx = np.random.randint(0, 2, fft_size)

        # Передача
        signal_td = tx.transmit(bits_tx)

        # Канал
        signal_rx_td = channel.add_noise(signal_td)

        # Приём
        symbols_rx = rx.fft(signal_rx_td)
        bits_rx = rx.demap(symbols_rx)

        # Подсчёт ошибок
        errors = np.sum(bits_tx != bits_rx)
        total_bit_errors += errors
        total_blocks += 1
        if errors > 0:
            block_errors += 1

        # Доля успешных битов в текущем опыте
        success_rate = 1.0 - errors / fft_size
        throughput_per_trial.append(success_rate)

    ber = total_bit_errors / (fft_size * num_trials)
    bler = block_errors / num_trials
    return ber, bler, throughput_per_trial


# ------------------------------ Основной скрипт ------------------------------
if __name__ == "__main__":
    # Параметры симуляции
    FFT_SIZE = 64
    SNR_DB_LIST = np.arange(0, 11, 1)      # от 0 до 10 дБ с шагом 1
    NUM_TRIALS = 1000

    # Запуск симуляции
    results = {}
    for snr in SNR_DB_LIST:
        print(f"Симуляция SNR = {snr} дБ...")
        ber, bler, thr_vals = simulate_snr(snr, FFT_SIZE, NUM_TRIALS)
        results[snr] = {
            'ber': ber,
            'bler': bler,
            'throughput_vals': thr_vals
        }
        print(f"  BER = {ber:.5f}, BLER = {bler:.5f}")

    # Теоретические кривые
    snr_lin_vals = 10 ** (np.array(SNR_DB_LIST) / 10.0)
    ber_theory = ber_awgn_theoretical(snr_lin_vals)
    bler_theory = bler_theoretical(FFT_SIZE, snr_lin_vals)
    shannon_capacity = np.log2(1 + snr_lin_vals)  # бит/с/Гц

    # ------------------------------ Построение графика BLER vs SNR ------------------------------
    plt.figure(figsize=(8, 6))
    plt.semilogy(SNR_DB_LIST, [results[s]['bler'] for s in SNR_DB_LIST], 'bo-', label='Симуляционный BLER')
    plt.semilogy(SNR_DB_LIST, bler_theory, 'r--', label='Теоретический BLER (на основе BER)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('BLER')
    plt.title('Вероятность ошибки в блоке (BLER)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Сохраняем первый график
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"plots_{date_str}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "BLER_vs_SNR.png"), dpi=150)
    plt.close()

    # ------------------------------ Построение графика Throughput vs CDF ------------------------------
    # Выбираем 4 характерных значения SNR для наглядности
    selected_snrs = [0, 3, 6, 10]
    # Для выбранных SNR строим эмпирические CDF
    plt.figure(figsize=(8, 6))
    for snr in selected_snrs:
        if snr not in results:
            continue
        thr_vals = np.array(results[snr]['throughput_vals'])
        # Сортировка для CDF
        thr_sorted = np.sort(thr_vals)
        cdf = np.arange(1, len(thr_sorted) + 1) / len(thr_sorted)
        plt.step(thr_sorted, cdf, label=f'SNR = {snr} дБ', where='post')
    plt.xlabel('Throughput (доля успешных битов)')
    plt.ylabel('CDF')
    plt.title('Эмпирическая функция распределения пропускной способности')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Throughput_CDF.png"), dpi=150)
    plt.close()

    # ------------------------------ Дополнительно: Throughput vs SNR с емкостью Шеннона ------------------------------
    # (не требовалось, но полезно для понимания)
    plt.figure(figsize=(8, 6))
    avg_throughput = [1 - results[s]['ber'] for s in SNR_DB_LIST]
    plt.plot(SNR_DB_LIST, avg_throughput, 'bo-', label='Система BPSK-OFDM (1 - BER)')
    plt.plot(SNR_DB_LIST, shannon_capacity, 'r--', label='Ёмкость Шеннона (бит/с/Гц)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('Throughput, бит/с/Гц')
    plt.title('Средняя пропускная способность')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Throughput_vs_SNR.png"), dpi=150)
    plt.close()

    # ------------------------------ Сохранение данных в .mat ------------------------------
    # Подготовка данных для экспорта
    mat_data = {
        'snr_db': SNR_DB_LIST,
        'bler_sim': np.array([results[s]['bler'] for s in SNR_DB_LIST]),
        'bler_theory': bler_theory,
        'ber_sim': np.array([results[s]['ber'] for s in SNR_DB_LIST]),
        'ber_theory': ber_theory,
        'throughput_cdf': {},   # для каждого SNR сохраняем значения и CDF
    }
    for snr in selected_snrs:
        if snr in results:
            thr_vals = np.array(results[snr]['throughput_vals'])
            thr_sorted = np.sort(thr_vals)
            cdf = np.arange(1, len(thr_sorted) + 1) / len(thr_sorted)
            mat_data[f'throughput_vals_snr{snr}'] = thr_vals
            mat_data[f'throughput_sorted_snr{snr}'] = thr_sorted
            mat_data[f'cdf_snr{snr}'] = cdf

    # Добавим средний throughput и шенноновскую ёмкость
    mat_data['avg_throughput'] = avg_throughput
    mat_data['shannon_capacity'] = shannon_capacity

    savemat(os.path.join(save_dir, 'simulation_data.mat'), mat_data)
    print(f"\nРезультаты сохранены в папку: {save_dir}")