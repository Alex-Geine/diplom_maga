import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
import os
import datetime

# ----------------------------- Таблица CP (в отсчетах) -----------------------------
# Для разных размеров FFT: normal CP (символы 1-6,8-13) и extended CP (символы 0,7)
cp_table = {
    512:  {'normal': 36, 'extended': 40},
    1024: {'normal': 72, 'extended': 80},
    2048: {'normal': 144, 'extended': 160},
    4096: {'normal': 288, 'extended': 320}
}

def get_cp_length(fft_size, cp_type='normal'):
    """Возвращает длину CP в отсчётах для заданного FFT."""
    if fft_size not in cp_table:
        raise ValueError(f"Размер FFT {fft_size} не поддерживается. Допустимы: {list(cp_table.keys())}")
    return cp_table[fft_size][cp_type]

# ----------------------------- Таблица RB для 5G NTN -----------------------------
ls_rb_table = {
    5:  {15: 25,  30: 11},
    10: {15: 52,  30: 24},
    15: {15: 79,  30: 38},
    20: {15: 106, 30: 51}
}
ka_rb_table = {
    50:  {60: 66,  120: 32},
    100: {60: 132, 120: 66},
    200: {60: 264, 120: 132},
    400: {120: 264}
}

def get_rb(band, bw_mhz, scs_khz):
    if band == 'L_S':
        return ls_rb_table.get(bw_mhz, {}).get(scs_khz)
    elif band == 'Ka':
        return ka_rb_table.get(bw_mhz, {}).get(scs_khz)
    return None

def get_fft_size_from_re(num_re):
    """Минимальная степень двойки > num_re (не менее 64)"""
    n = 64
    while n <= num_re:
        n *= 2
    return n

# ----------------------------- Классы с поддержкой CP -----------------------------
class OFDMTx:
    def __init__(self, num_re, fft_size, cp_type='normal'):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2

    def map(self, bits):
        return 2 * bits - 1

    def ifft(self, symbols_re):
        freq = np.zeros(self.fft_size, dtype=complex)
        freq[self.offset:self.offset + self.num_re] = symbols_re
        return np.fft.ifft(freq, norm='ortho')

    def add_cp(self, time_signal):
        """Добавить циклический префикс: последние cp_len отсчётов в начало"""
        return np.concatenate([time_signal[-self.cp_len:], time_signal])

    def transmit(self, bits):
        symbols = self.map(bits)
        ofdm_symbol = self.ifft(symbols)
        return self.add_cp(ofdm_symbol)


class OFDMRx:
    def __init__(self, num_re, fft_size, cp_type='normal'):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2

    def remove_cp(self, rx_signal):
        """Удалить циклический префикс"""
        return rx_signal[self.cp_len:self.cp_len + self.fft_size]

    def fft(self, time_signal):
        return np.fft.fft(time_signal, norm='ortho')

    def extract_re(self, freq_domain):
        return freq_domain[self.offset:self.offset + self.num_re]

    def demap(self, symbols):
        return (np.real(symbols) > 0).astype(int)

    def receive(self, rx_signal_with_cp):
        signal_no_cp = self.remove_cp(rx_signal_with_cp)
        freq = self.fft(signal_no_cp)
        re_symbols = self.extract_re(freq)
        return self.demap(re_symbols)


class Channel:
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.snr_lin = 10 ** (snr_db / 10.0)
        self.N0 = 1.0 / self.snr_lin

    def add_noise(self, signal):
        noise = np.sqrt(self.N0/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise


# ----------------------------- Теоретические функции -----------------------------
def ber_awgn(snr_lin):
    return 0.5 * erfc(np.sqrt(snr_lin))

def bler_theoretical(num_bits, snr_lin):
    ber = ber_awgn(snr_lin)
    return 1 - (1 - ber) ** num_bits

def shannon_capacity(snr_lin):
    return np.log2(1 + snr_lin)


# ----------------------------- Симуляция одного SNR -----------------------------
def simulate_snr(snr_db, tx, rx, channel, num_trials):
    total_bit_errors = 0
    block_errors = 0

    for _ in range(num_trials):
        bits_tx = np.random.randint(0, 2, tx.num_re)
        signal_td = tx.transmit(bits_tx)           # с CP
        signal_rx = channel.add_noise(signal_td)   # с CP
        bits_rx = rx.receive(signal_rx)            # внутри удаляется CP
        errors = np.sum(bits_tx != bits_rx)
        total_bit_errors += errors
        if errors > 0:
            block_errors += 1

    ber = total_bit_errors / (tx.num_re * num_trials)
    bler = block_errors / num_trials
    # Спектральная эффективность с учётом CP:
    # На один OFDM символ приходится fft_size + cp_len отсчётов,
    # из них полезные отсчёты (данные) только fft_size. Полезные поднесущие: num_re.
    # Throughput = (1 - BER) * (num_re полезных бит) / (общее число отсчётов)
    #             = (1 - BER) * (num_re) / (fft_size + cp_len)
    throughput = (1 - ber) * tx.num_re / (tx.fft_size + tx.cp_len)
    return ber, bler, throughput


# ----------------------------- Основной скрипт -----------------------------
if __name__ == "__main__":
    # Конфигурация 5G NTN
    BAND = 'L_S'           # 'L_S' или 'Ka'
    BW_MHZ = 10            # МГц
    SCS_KHZ = 30           # кГц

    rb = get_rb(BAND, BW_MHZ, SCS_KHZ)
    if rb is None:
        print(f"Ошибка: нет RB для {BAND} BW={BW_MHZ} МГц SCS={SCS_KHZ} кГц")
        exit(1)

    num_re = rb * 12
    fft_size = get_fft_size_from_re(num_re)
    cp_type = 'normal'      # используем стандартный CP для всех символов

    print(f"Конфигурация: {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц")
    print(f"  → RB = {rb}, RE = {num_re}")
    print(f"  → FFT size = {fft_size}, CP length = {get_cp_length(fft_size, cp_type)} отсчётов")
    print(f"  → Защитные поднесущие: {(fft_size - num_re)//2} слева и справа")

    SNR_DB_LIST = np.arange(0, 11, 1)    # 0..10 дБ
    NUM_TRIALS = 1000

    tx = OFDMTx(num_re, fft_size, cp_type)
    rx = OFDMRx(num_re, fft_size, cp_type)

    results = {}
    for snr in SNR_DB_LIST:
        print(f"Симуляция SNR = {snr} дБ ...")
        channel = Channel(snr)
        ber, bler, thr = simulate_snr(snr, tx, rx, channel, NUM_TRIALS)
        results[snr] = {'ber': ber, 'bler': bler, 'throughput': thr}
        print(f"  BER = {ber:.5f}, BLER = {bler:.5f}, Throughput = {thr:.6f} бит/с/Гц")

    # Теоретические кривые
    snr_lin_vals = 10 ** (np.array(SNR_DB_LIST) / 10.0)
    bler_theory = bler_theoretical(num_re, snr_lin_vals)
    shannon = shannon_capacity(snr_lin_vals)

    # Извлекаем результаты
    snr_list = list(results.keys())
    bler_sim = [results[s]['bler'] for s in snr_list]
    thr_sim = [results[s]['throughput'] for s in snr_list]

    # Построение BLER vs SNR
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_list, bler_sim, 'bo-', label='Симуляция (с CP)')
    plt.semilogy(snr_list, bler_theory, 'r--', label='Теория (независимые биты)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('BLER')
    plt.title(f'BLER для {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц\n'
              f'RE = {num_re}, FFT = {fft_size}, CP = {tx.cp_len} отсчётов')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    bler_fig = plt.gcf()

    # Throughput vs SNR + Шеннон
    plt.figure(figsize=(8, 6))
    plt.plot(snr_list, thr_sim, 'bo-', label='BPSK-OFDM с CP')
    plt.plot(snr_list, shannon, 'r--', label='Ёмкость Шеннона (бит/с/Гц)')
    # Также можно нарисовать верхнюю границу без учёта BER: (1 - 0) * num_re/(fft_size+cp_len)
    max_thr = num_re / (fft_size + tx.cp_len)
    plt.axhline(y=max_thr, color='gray', linestyle=':', label=f'Макс. спектр. эффективность = {max_thr:.3f}')
    plt.xlabel('SNR, дБ')
    plt.ylabel('Throughput, бит/с/Гц')
    plt.title(f'Пропускная способность с учётом CP\n'
              f'RE/FFT = {num_re}/{fft_size}, CP = {tx.cp_len} отсч. → фактор {max_thr:.3f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    thr_fig = plt.gcf()

    # Сохранение
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"NTN_CP_{BAND}_{BW_MHZ}MHz_{SCS_KHZ}kHz_{date_str}"
    os.makedirs(save_dir, exist_ok=True)

    bler_fig.savefig(os.path.join(save_dir, "BLER_vs_SNR.png"), dpi=150)
    thr_fig.savefig(os.path.join(save_dir, "Throughput_vs_SNR.png"), dpi=150)
    plt.close('all')

    # Данные для MATLAB
    mat_data = {
        'config': {
            'band': BAND,
            'bw_mhz': BW_MHZ,
            'scs_khz': SCS_KHZ,
            'num_rb': rb,
            'num_re': num_re,
            'fft_size': fft_size,
            'cp_length': tx.cp_len,
            'max_spectral_efficiency': max_thr
        },
        'snr_db': np.array(snr_list),
        'bler_sim': np.array(bler_sim),
        'bler_theory': bler_theory,
        'throughput_sim': np.array(thr_sim),
        'shannon_capacity': shannon
    }
    savemat(os.path.join(save_dir, 'simulation_data.mat'), mat_data)

    print(f"\nРезультаты сохранены в папку: {save_dir}")
    print("Файлы: BLER_vs_SNR.png, Throughput_vs_SNR.png, simulation_data.mat")