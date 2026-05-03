import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
import os
import datetime

# ------------------------------ Конфигурация 5G NR NTN (таблица) ------------------------------
# Структура: для каждого диапазона, полосы и SCS -> количество RB
# L/S диапазоны (n255/n256), SCS 15 и 30 кГц
ls_config = {
    5:  {15: 25,  30: 11},
    10: {15: 52,  30: 24},
    15: {15: 79,  30: 38},
    20: {15: 106, 30: 51}
}
# Ka диапазон (n510/511/512), SCS 60 и 120 кГц
ka_config = {
    50:  {60: 66,  120: 32},
    100: {60: 132, 120: 66},
    200: {60: 264, 120: 132},
    400: {120: 264}   # для 400 МГц и SCS 60 не определено по 3GPP
}

def get_fft_size(band, bw_mhz, scs_khz):
    """
    Возвращает FFT size = количество поднесущих = RB * 12.
    band: 'L_S' или 'Ka'
    bw_mhz: полоса в МГц (5,10,15,20 для L_S; 50,100,200,400 для Ka)
    scs_khz: 15,30,60,120
    """
    if band == 'L_S':
        rb = ls_config.get(bw_mhz, {}).get(scs_khz)
    elif band == 'Ka':
        rb = ka_config.get(bw_mhz, {}).get(scs_khz)
    else:
        raise ValueError("Неподдерживаемый диапазон. Используйте 'L_S' или 'Ka'")
    if rb is None:
        raise ValueError(f"Для {band} BW={bw_mhz} МГц, SCS={scs_khz} кГц нет данных в таблице 3GPP")
    return rb * 12   # число поднесущих (RE на один OFDM символ)

# ------------------------------ Классы ------------------------------
class OFDMTx:
    """OFDM передатчик с BPSK и IFFT"""
    def __init__(self, fft_size):
        self.fft_size = fft_size

    def map(self, bits):
        """BPSK: 0 -> -1, 1 -> +1"""
        return 2 * bits - 1

    def ifft(self, symbols):
        """Обратное БПФ с ортонормировкой"""
        return np.fft.ifft(symbols, norm='ortho')

    def transmit(self, bits):
        """биты -> символы -> временной сигнал"""
        symbols = self.map(bits)
        signal = self.ifft(symbols)
        return signal


class OFDMRx:
    """OFDM приёмник с FFT и демодуляцией BPSK"""
    def __init__(self, fft_size):
        self.fft_size = fft_size

    def fft(self, signal):
        return np.fft.fft(signal, norm='ortho')

    def demap(self, symbols):
        return (np.real(symbols) > 0).astype(int)


class Channel:
    """Плоский канал с AWGN"""
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.snr_lin = 10 ** (snr_db / 10.0)
        self.N0 = 1.0 / self.snr_lin   # дисперсия шума на комплексную выборку

    def add_noise(self, signal):
        noise = np.sqrt(self.N0 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise


# ------------------------------ Теоретические функции ------------------------------
def ber_awgn(snr_lin):
    return 0.5 * erfc(np.sqrt(snr_lin))

def bler_theoretical(fft_size, snr_lin):
    ber = ber_awgn(snr_lin)
    return 1 - (1 - ber) ** fft_size

def shannon_capacity(snr_lin):
    return np.log2(1 + snr_lin)   # бит/с/Гц


# ------------------------------ Симуляция для одного SNR ------------------------------
def simulate_snr(snr_db, fft_size, num_trials):
    tx = OFDMTx(fft_size)
    rx = OFDMRx(fft_size)
    channel = Channel(snr_db)

    total_bit_errors = 0
    block_errors = 0

    for _ in range(num_trials):
        bits_tx = np.random.randint(0, 2, fft_size)
        signal_td = tx.transmit(bits_tx)
        signal_rx_td = channel.add_noise(signal_td)
        symbols_rx = rx.fft(signal_rx_td)
        bits_rx = rx.demap(symbols_rx)

        errors = np.sum(bits_tx != bits_rx)
        total_bit_errors += errors
        if errors > 0:
            block_errors += 1

    ber = total_bit_errors / (fft_size * num_trials)
    bler = block_errors / num_trials
    throughput = 1 - ber   # бит/с/Гц для BPSK
    return ber, bler, throughput


# ------------------------------ Основной скрипт ------------------------------
if __name__ == "__main__":
    # ------------------ Выбор конфигурации 5G NTN ------------------
    # Параметры (можно менять)
    BAND = 'L_S'          # 'L_S' или 'Ka'
    BW_MHZ = 10           # для L_S: 5,10,15,20; для Ka: 50,100,200,400
    SCS_KHZ = 30          # для L_S: 15 или 30; для Ka: 60 или 120

    # Вычисляем FFT size на основе таблицы
    try:
        FFT_SIZE = get_fft_size(BAND, BW_MHZ, SCS_KHZ)
        print(f"Конфигурация: {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц → FFT size = {FFT_SIZE} поднесущих")
    except Exception as e:
        print(f"Ошибка: {e}")
        exit(1)

    # Параметры симуляции
    SNR_DB_LIST = np.arange(0, 11, 1)   # 0..10 дБ
    NUM_TRIALS = 1000

    # Запуск симуляции
    results = {}
    for snr in SNR_DB_LIST:
        print(f"Симуляция SNR = {snr} дБ...")
        ber, bler, thr = simulate_snr(snr, FFT_SIZE, NUM_TRIALS)
        results[snr] = {'ber': ber, 'bler': bler, 'throughput': thr}
        print(f"  BER = {ber:.5f}, BLER = {bler:.5f}, Throughput = {thr:.5f} бит/с/Гц")

    # Теоретические кривые
    snr_lin_vals = 10 ** (np.array(SNR_DB_LIST) / 10.0)
    bler_theory = bler_theoretical(FFT_SIZE, snr_lin_vals)
    capacity_theory = shannon_capacity(snr_lin_vals)

    # Извлекаем симуляционные значения
    bler_sim = [results[s]['bler'] for s in SNR_DB_LIST]
    throughput_sim = [results[s]['throughput'] for s in SNR_DB_LIST]

    # ------------------------------ Построение BLER vs SNR ------------------------------
    plt.figure(figsize=(8, 6))
    plt.semilogy(SNR_DB_LIST, bler_sim, 'bo-', label='Симуляция BLER')
    plt.semilogy(SNR_DB_LIST, bler_theory, 'r--', label='Теоретический BLER (BPSK, AWGN)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('BLER')
    plt.title(f'Вероятность ошибки в блоке (BLER) для {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # ------------------------------ Построение Throughput vs SNR ------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(SNR_DB_LIST, throughput_sim, 'bo-', label='Симуляционный throughput (1-BER)')
    plt.plot(SNR_DB_LIST, capacity_theory, 'r--', label='Ёмкость Шеннона (бит/с/Гц)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('Throughput, бит/с/Гц')
    plt.title(f'Пропускная способность для {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # ------------------------------ Сохранение результатов ------------------------------
    # Создаём папку с датой и временем
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"NTN_results_{BAND}_{BW_MHZ}MHz_{SCS_KHZ}kHz_{date_str}"
    os.makedirs(save_dir, exist_ok=True)

    # Сохраняем графики
    plt.figure(1)
    plt.savefig(os.path.join(save_dir, "BLER_vs_SNR.png"), dpi=150)
    plt.figure(2)
    plt.savefig(os.path.join(save_dir, "Throughput_vs_SNR.png"), dpi=150)
    plt.close('all')

    # Сохраняем данные в .mat
    mat_data = {
        'config_band': BAND,
        'config_bw_mhz': BW_MHZ,
        'config_scs_khz': SCS_KHZ,
        'fft_size': FFT_SIZE,
        'snr_db': SNR_DB_LIST,
        'bler_simulation': np.array(bler_sim),
        'bler_theory': bler_theory,
        'throughput_simulation': np.array(throughput_sim),
        'shannon_capacity': capacity_theory
    }
    savemat(os.path.join(save_dir, 'simulation_data.mat'), mat_data)

    print(f"\nРезультаты сохранены в папку: {save_dir}")
    print("Содержит: BLER_vs_SNR.png, Throughput_vs_SNR.png, simulation_data.mat")