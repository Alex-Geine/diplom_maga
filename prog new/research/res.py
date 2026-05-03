import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
import os
import datetime

# ------------------------------ Конфигурация 5G NR NTN (таблица RB) ------------------------------
# L/S диапазоны (n255/n256)
ls_rb_table = {
    5:  {15: 25,  30: 11},
    10: {15: 52,  30: 24},
    15: {15: 79,  30: 38},
    20: {15: 106, 30: 51}
}
# Ka диапазон (n510/511/512)
ka_rb_table = {
    50:  {60: 66,  120: 32},
    100: {60: 132, 120: 66},
    200: {60: 264, 120: 132},
    400: {120: 264}   # для 400 МГц SCS=60 кГц не определено
}

def get_rb(band, bw_mhz, scs_khz):
    """Возвращает количество RB по таблице 3GPP"""
    if band == 'L_S':
        return ls_rb_table.get(bw_mhz, {}).get(scs_khz)
    elif band == 'Ka':
        return ka_rb_table.get(bw_mhz, {}).get(scs_khz)
    else:
        raise ValueError("band должен быть 'L_S' или 'Ka'")

def get_fft_size_from_re(num_re):
    """Выбор минимальной степени двойки > num_re, но не менее 64"""
    n = 64
    while n <= num_re:
        n *= 2
    return n

# ------------------------------ Классы ------------------------------
class OFDMTx:
    """OFDM передатчик с BPSK, центрированием поднесущих и IFFT"""
    def __init__(self, num_re, fft_size):
        self.num_re = num_re          # количество используемых поднесущих
        self.fft_size = fft_size      # размер БПФ (степень двойки)
        # Смещение: размещаем RE в центре
        self.offset = (fft_size - num_re) // 2

    def map(self, bits):
        """BPSK: 0 → -1, 1 → +1"""
        return 2 * bits - 1

    def ifft(self, symbols_re):
        """Размещение RE в центре, zero padding, IFFT с ортонормировкой"""
        freq_domain = np.zeros(self.fft_size, dtype=complex)
        freq_domain[self.offset:self.offset + self.num_re] = symbols_re
        return np.fft.ifft(freq_domain, norm='ortho')

    def transmit(self, bits):
        """Биты → BPSK-символы → разнесение по частоте → IFFT"""
        symbols = self.map(bits)
        return self.ifft(symbols)


class OFDMRx:
    """OFDM приёмник: FFT, извлечение центральных поднесущих, BPSK демодуляция"""
    def __init__(self, num_re, fft_size):
        self.num_re = num_re
        self.fft_size = fft_size
        self.offset = (fft_size - num_re) // 2

    def fft(self, time_signal):
        """Прямое БПФ с ортонормировкой"""
        return np.fft.fft(time_signal, norm='ortho')

    def extract_re(self, freq_domain):
        """Извлечение только центральных поднесущих, где были данные"""
        return freq_domain[self.offset:self.offset + self.num_re]

    def demap(self, symbols):
        """Жёсткое решение BPSK: real > 0 → 1, иначе 0"""
        return (np.real(symbols) > 0).astype(int)

    def receive(self, time_signal):
        """Полный цикл приёма: FFT → извлечение RE → демодуляция"""
        fd = self.fft(time_signal)
        re_symbols = self.extract_re(fd)
        return self.demap(re_symbols)


class Channel:
    """Плоский канал с аддитивным белым гауссовым шумом (AWGN)"""
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.snr_lin = 10 ** (snr_db / 10.0)
        self.N0 = 1.0 / self.snr_lin   # дисперсия комплексного шума на выборку

    def add_noise(self, signal):
        """Добавить комплексный белый шум"""
        noise = np.sqrt(self.N0/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise


# ------------------------------ Теоретические кривые ------------------------------
def ber_awgn(snr_lin):
    """Теоретическая BER для BPSK в AWGN"""
    return 0.5 * erfc(np.sqrt(snr_lin))

def bler_theoretical(num_bits_per_block, snr_lin):
    """BLER для блока независимых битов (идеальное перемежение)"""
    ber = ber_awgn(snr_lin)
    return 1 - (1 - ber) ** num_bits_per_block

def shannon_capacity(snr_lin):
    """Ёмкость Шеннона для комплексного канала (бит/с/Гц)"""
    return np.log2(1 + snr_lin)


# ------------------------------ Симуляция для одного SNR ------------------------------
def simulate_snr(snr_db, tx, rx, channel, num_trials):
    """Симуляция для заданного SNR и готовых объектов передатчика/приёмника"""
    total_bit_errors = 0
    block_errors = 0

    for _ in range(num_trials):
        bits_tx = np.random.randint(0, 2, tx.num_re)
        signal_td = tx.transmit(bits_tx)
        signal_rx_td = channel.add_noise(signal_td)
        bits_rx = rx.receive(signal_rx_td)

        errors = np.sum(bits_tx != bits_rx)
        total_bit_errors += errors
        if errors > 0:
            block_errors += 1

    ber = total_bit_errors / (tx.num_re * num_trials)
    bler = block_errors / num_trials
    # Спектральная эффективность: (1 - BER) * (количество бит на символ) * (RE/FFT)
    # Для BPSK бит/символ = 1. Умножаем на долю занятых поднесущих.
    throughput = (1 - ber) * (tx.num_re / tx.fft_size)   # бит/с/Гц
    return ber, bler, throughput


# ------------------------------ Основной скрипт ------------------------------
if __name__ == "__main__":
    # ------------------------- Выбор конфигурации 5G NTN -------------------------
    BAND = 'L_S'          # 'L_S' или 'Ka'
    BW_MHZ = 10           # МГц (см. таблицу)
    SCS_KHZ = 30          # кГц (см. таблицу)

    # Получаем количество RB из таблицы 3GPP
    rb = get_rb(BAND, BW_MHZ, SCS_KHZ)
    if rb is None:
        print(f"Ошибка: нет данных для {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц")
        exit(1)

    num_re = rb * 12            # число ресурсных элементов (поднесущих) на один OFDM-символ
    fft_size = get_fft_size_from_re(num_re)   # физический размер БПФ (степень двойки)

    print(f"Конфигурация: {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц")
    print(f"  → RB = {rb}, RE = {num_re}")
    print(f"  → FFT size = {fft_size} (ближайшая степень двойки > {num_re})")
    print(f"  → Защитные поднесущие: {(fft_size - num_re)//2} слева и справа")

    # Параметры симуляции
    SNR_DB_LIST = np.arange(0, 11, 1)   # 0...10 дБ
    NUM_TRIALS = 1000

    # Создаём передатчик и приёмник (одинаковы для всех SNR)
    tx = OFDMTx(num_re, fft_size)
    rx = OFDMRx(num_re, fft_size)

    # Запуск по всем SNR
    results = {}
    for snr in SNR_DB_LIST:
        print(f"\nСимуляция SNR = {snr} дБ ...")
        channel = Channel(snr)
        ber, bler, thr = simulate_snr(snr, tx, rx, channel, NUM_TRIALS)
        results[snr] = {'ber': ber, 'bler': bler, 'throughput': thr}
        print(f"  BER = {ber:.5f}, BLER = {bler:.5f}, Throughput = {thr:.4f} бит/с/Гц")

    # Теоретические кривые (на основе SNR на поднесущую)
    snr_lin_vals = 10 ** (np.array(SNR_DB_LIST) / 10.0)
    bler_theory = bler_theoretical(num_re, snr_lin_vals)   # блок из num_re бит
    shannon = shannon_capacity(snr_lin_vals)

    # Извлекаем симуляционные значения
    snr_list = list(results.keys())
    ber_sim = [results[s]['ber'] for s in snr_list]
    bler_sim = [results[s]['bler'] for s in snr_list]
    throughput_sim = [results[s]['throughput'] for s in snr_list]

    # ------------------------------ Построение графиков ------------------------------
    # 1) BLER vs SNR
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_list, bler_sim, 'bo-', label='Симуляция')
    plt.semilogy(snr_list, bler_theory, 'r--', label='Теория (независимые биты)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('BLER')
    plt.title(f'BLER для {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц\n'
              f'RE = {num_re}, FFT = {fft_size}')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    bler_fig = plt.gcf()

    # 2) Throughput vs SNR (сравнение с ёмкостью Шеннона)
    plt.figure(figsize=(8, 6))
    plt.plot(snr_list, throughput_sim, 'bo-', label='BPSK-OFDM (центрированные RE)')
    plt.plot(snr_list, shannon, 'r--', label='Ёмкость Шеннона (бит/с/Гц)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('Throughput, бит/с/Гц')
    plt.title(f'Пропускная способность для {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц\n'
              f'RE/FFT = {num_re}/{fft_size} = {num_re/fft_size:.2f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    thr_fig = plt.gcf()

    # ------------------------------ Сохранение ------------------------------
    # Папка с датой и временем + параметрами
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"NTN_{BAND}_{BW_MHZ}MHz_{SCS_KHZ}kHz_{date_str}"
    os.makedirs(save_dir, exist_ok=True)

    bler_fig.savefig(os.path.join(save_dir, "BLER_vs_SNR.png"), dpi=150)
    thr_fig.savefig(os.path.join(save_dir, "Throughput_vs_SNR.png"), dpi=150)
    plt.close('all')

    # Сохраняем данные в .mat
    mat_data = {
        'config': {
            'band': BAND,
            'bw_mhz': BW_MHZ,
            'scs_khz': SCS_KHZ,
            'num_rb': rb,
            'num_re': num_re,
            'fft_size': fft_size,
        },
        'snr_db': np.array(snr_list),
        'ber_sim': np.array(ber_sim),
        'bler_sim': np.array(bler_sim),
        'bler_theory': bler_theory,
        'throughput_sim': np.array(throughput_sim),
        'shannon_capacity': shannon,
    }
    savemat(os.path.join(save_dir, 'simulation_data.mat'), mat_data)

    print(f"\nРезультаты сохранены в папку: {save_dir}")
    print("Файлы: BLER_vs_SNR.png, Throughput_vs_SNR.png, simulation_data.mat")