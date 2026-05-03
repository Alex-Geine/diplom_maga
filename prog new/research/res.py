import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
import os
import datetime

# ----------------------------- Таблица CP (в отсчётах) -----------------------------
cp_table = {
    512:  {'normal': 36, 'extended': 40},
    1024: {'normal': 72, 'extended': 80},
    2048: {'normal': 144, 'extended': 160},
    4096: {'normal': 288, 'extended': 320}
}

def get_cp_length(fft_size, cp_type='normal'):
    if fft_size not in cp_table:
        raise ValueError(f"FFT size {fft_size} not supported. Allowed: {list(cp_table.keys())}")
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
    n = 64
    while n <= num_re:
        n *= 2
    return n

# ----------------------------- Модуляция и демодуляция -----------------------------
def modulate(bits, mod_type):
    """Отображение бит в комплексные символы (средняя мощность = 1)"""
    bits = np.asarray(bits)
    if mod_type == 'BPSK':
        return 2 * bits - 1
    elif mod_type == 'QPSK':
        # QPSK: пары бит -> (1-2b0) + j(1-2b1) / sqrt(2) -> мощность 1
        even = bits[0::2]
        odd = bits[1::2]
        I = 1 - 2 * even
        Q = 1 - 2 * odd
        symbols = (I + 1j * Q) / np.sqrt(2)
        return symbols
    elif mod_type == '16QAM':
        # 16QAM (Gray mapping), нормировка на среднюю мощность 1
        # Разбиваем на группы по 4 бита: b0 b1 b2 b3 -> I = (1-2b0)*(2 - (1-2b1)), Q аналогично
        bits_r = bits.reshape(-1, 4)
        b0, b1, b2, b3 = bits_r[:,0], bits_r[:,1], bits_r[:,2], bits_r[:,3]
        I = (1 - 2*b0) * (2 - (1 - 2*b1))
        Q = (1 - 2*b2) * (2 - (1 - 2*b3))
        symbols = (I + 1j * Q) / np.sqrt(10)  # нормировка: средняя мощность = 1
        return symbols
    elif mod_type == '64QAM':
        bits_r = bits.reshape(-1, 6)
        b0,b1,b2,b3,b4,b5 = bits_r[:,0], bits_r[:,1], bits_r[:,2], bits_r[:,3], bits_r[:,4], bits_r[:,5]
        I = (1-2*b0) * (4 - (1-2*b1)*(2 - (1-2*b2)))
        Q = (1-2*b3) * (4 - (1-2*b4)*(2 - (1-2*b5)))
        symbols = (I + 1j*Q) / np.sqrt(42)  # средняя мощность = 1
        return symbols
    elif mod_type == '256QAM':
        bits_r = bits.reshape(-1, 8)
        # Для простоты используем формулу: I = (1-2b0)*(8 - (1-2b1)*(4 - (1-2b2)*(2 - (1-2b3))))
        b = [bits_r[:,i] for i in range(8)]
        I = (1-2*b[0]) * (8 - (1-2*b[1])*(4 - (1-2*b[2])*(2 - (1-2*b[3]))))
        Q = (1-2*b[4]) * (8 - (1-2*b[5])*(4 - (1-2*b[6])*(2 - (1-2*b[7]))))
        symbols = (I + 1j*Q) / np.sqrt(170)  # средняя мощность = 1
        return symbols
    else:
        raise ValueError(f"Неизвестная модуляция: {mod_type}")

def demodulate(symbols, mod_type):
    """Жёсткая демодуляция (решение по правилу максимального правдоподобия)"""
    if mod_type == 'BPSK':
        return (np.real(symbols) > 0).astype(int)
    elif mod_type == 'QPSK':
        # Умножаем на sqrt(2) чтобы вернуться к решётке +/-1
        sym_scaled = symbols * np.sqrt(2)
        I = np.real(sym_scaled)
        Q = np.imag(sym_scaled)
        bits = np.empty(len(symbols)*2, dtype=int)
        bits[0::2] = (I < 0).astype(int)
        bits[1::2] = (Q < 0).astype(int)
        return bits
    elif mod_type == '16QAM':
        sym_scaled = symbols * np.sqrt(10)
        I = np.real(sym_scaled)
        Q = np.imag(sym_scaled)
        # Демодуляция: для I используем пороги -2,0,2 (решётка: -3,-1,1,3)
        # Решение методом slicing
        def slice_16(x):
            # x - вещественный, возвращает 2 бита
            if x < -2:
                return (1,1)  # -3
            elif x < 0:
                return (1,0)  # -1
            elif x < 2:
                return (0,1)  # 1
            else:
                return (0,0)  # 3
        bits = np.empty(len(symbols)*4, dtype=int)
        for i, (i_val, q_val) in enumerate(zip(I, Q)):
            b0,b1 = slice_16(i_val)
            b2,b3 = slice_16(q_val)
            bits[4*i:4*i+4] = [b0,b1,b2,b3]
        return bits
    elif mod_type == '64QAM':
        sym_scaled = symbols * np.sqrt(42)
        I = np.real(sym_scaled)
        Q = np.imag(sym_scaled)
        # Решётка: -7,-5,-3,-1,1,3,5,7 -> пороги -6,-4,-2,0,2,4,6
        def slice_64(x):
            if x < -6:
                return (1,1,1)
            elif x < -4:
                return (1,1,0)
            elif x < -2:
                return (1,0,1)
            elif x < 0:
                return (1,0,0)
            elif x < 2:
                return (0,1,1)
            elif x < 4:
                return (0,1,0)
            elif x < 6:
                return (0,0,1)
            else:
                return (0,0,0)
        bits = np.empty(len(symbols)*6, dtype=int)
        for i, (i_val, q_val) in enumerate(zip(I, Q)):
            b0,b1,b2 = slice_64(i_val)
            b3,b4,b5 = slice_64(q_val)
            bits[6*i:6*i+6] = [b0,b1,b2,b3,b4,b5]
        return bits
    elif mod_type == '256QAM':
        sym_scaled = symbols * np.sqrt(170)
        I = np.real(sym_scaled)
        Q = np.imag(sym_scaled)
        # Пороги для 256QAM: -14,-12,...,12,14 (упростим - используем общий подход)
        # Для простоты используем битовый поиск, но здесь сделаем линейно
        levels = np.arange(-15, 16, 2)  # -15,-13,...,15
        thresholds = np.arange(-14, 15, 2)
        def slice_256(x):
            idx = np.digitize(x, thresholds, right=False)
            # idx от 0 до 15, преобразуем в 4 бита (Gray mapping обратно)
            # Упрощённо: прямое отображение битов (non-Gray для простоты)
            bits4 = [(idx>>3)&1, (idx>>2)&1, (idx>>1)&1, idx&1]
            return bits4
        bits = np.empty(len(symbols)*8, dtype=int)
        for i, (i_val, q_val) in enumerate(zip(I, Q)):
            i_bits = slice_256(i_val)
            q_bits = slice_256(q_val)
            bits[8*i:8*i+8] = i_bits + q_bits
        return bits
    else:
        raise ValueError(f"Неизвестная модуляция: {mod_type}")

def bits_per_symbol(mod_type):
    mapping = {'BPSK':1, 'QPSK':2, '16QAM':4, '64QAM':6, '256QAM':8}
    return mapping[mod_type]

# ----------------------------- Классы OFDM -----------------------------
class OFDMTx:
    def __init__(self, num_re, fft_size, cp_type='normal', mod_type='QPSK'):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2
        self.mod_type = mod_type
        self.bits_per_sym = bits_per_symbol(mod_type)

    def transmit(self, bits):
        # bits: битовый массив длины num_re * bits_per_sym
        symbols = modulate(bits, self.mod_type)
        freq = np.zeros(self.fft_size, dtype=complex)
        freq[self.offset:self.offset + self.num_re] = symbols
        time_signal = np.fft.ifft(freq, norm='ortho')
        # Добавляем CP
        return np.concatenate([time_signal[-self.cp_len:], time_signal])

class OFDMRx:
    def __init__(self, num_re, fft_size, cp_type='normal', mod_type='QPSK'):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2
        self.mod_type = mod_type
        self.bits_per_sym = bits_per_symbol(mod_type)

    def receive(self, rx_signal):
        # Удаляем CP
        signal = rx_signal[self.cp_len:self.cp_len + self.fft_size]
        freq = np.fft.fft(signal, norm='ortho')
        symbols = freq[self.offset:self.offset + self.num_re]
        bits = demodulate(symbols, self.mod_type)
        return bits

class Channel:
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.snr_lin = 10**(snr_db/10)
        self.N0 = 1.0 / self.snr_lin   # дисперсия шума на комплексную выборку

    def add_noise(self, signal):
        noise = np.sqrt(self.N0/2)*(np.random.randn(*signal.shape)+1j*np.random.randn(*signal.shape))
        return signal + noise

# ----------------------------- Теоретические BER -----------------------------
def ber_awgn_qam(snr_lin, mod_type):
    """Аппроксимация BER для M-QAM в AWGN (серое кодирование)"""
    M = 2**bits_per_symbol(mod_type)
    if M == 2:  # BPSK
        return 0.5 * erfc(np.sqrt(snr_lin))
    elif M == 4:  # QPSK
        return 0.5 * erfc(np.sqrt(snr_lin/2))  # уточнение для QPSK
    else:
        # Приближение для M-QAM: BER ≈ (4/log2(M)) * (1 - 1/sqrt(M)) * Q(sqrt(3*log2(M)*snr_lin/(M-1)))
        k = bits_per_symbol(mod_type)
        ber = (4/k) * (1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt(3*k*snr_lin/(2*(M-1))))
        return ber

def bler_theoretical(num_bits, snr_lin, mod_type):
    ber = ber_awgn_qam(snr_lin, mod_type)
    return 1 - (1 - ber) ** num_bits

def shannon_capacity(snr_lin):
    return np.log2(1 + snr_lin)

# ----------------------------- Симуляция -----------------------------
def simulate_snr(snr_db, tx, rx, channel, num_trials):
    total_bits_errors = 0
    block_errors = 0
    bits_per_ofdm = tx.num_re * tx.bits_per_sym

    for _ in range(num_trials):
        bits_tx = np.random.randint(0, 2, bits_per_ofdm)
        signal_td = tx.transmit(bits_tx)
        signal_rx = channel.add_noise(signal_td)
        bits_rx = rx.receive(signal_rx)
        errors = np.sum(bits_tx != bits_rx)
        total_bits_errors += errors
        if errors > 0:
            block_errors += 1
    ber = total_bits_errors / (bits_per_ofdm * num_trials)
    bler = block_errors / num_trials
    # Спектральная эффективность: (1 - BER) * (bits_per_sym * num_re) / (fft_size + cp_len)
    spectral_eff_max = (tx.bits_per_sym * tx.num_re) / (tx.fft_size + tx.cp_len)
    throughput = (1 - ber) * spectral_eff_max
    return ber, bler, throughput

# ----------------------------- Основной скрипт -----------------------------
if __name__ == "__main__":
    # Конфигурация
    BAND = 'L_S'
    BW_MHZ = 10
    SCS_KHZ = 30
    MODULATION = 'QPSK'   # BPSK, QPSK, 16QAM, 64QAM, 256QAM

    rb = get_rb(BAND, BW_MHZ, SCS_KHZ)
    if rb is None:
        print("Некорректные параметры")
        exit(1)

    num_re = rb * 12
    fft_size = get_fft_size_from_re(num_re)
    cp_type = 'normal'

    print(f"NTN конфигурация: {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц")
    print(f"  RE = {num_re}, FFT = {fft_size}, CP = {get_cp_length(fft_size, cp_type)} отсчётов")
    print(f"  Модуляция: {MODULATION} ({bits_per_symbol(MODULATION)} бит/символ)")

    SNR_DB_LIST = np.arange(0, 21, 2)   # от 0 до 20 дБ с шагом 2
    NUM_TRIALS = 2000

    tx = OFDMTx(num_re, fft_size, cp_type, MODULATION)
    rx = OFDMRx(num_re, fft_size, cp_type, MODULATION)

    results = {}
    for snr in SNR_DB_LIST:
        print(f"Симуляция SNR = {snr} дБ ...")
        chan = Channel(snr)
        ber, bler, thr = simulate_snr(snr, tx, rx, chan, NUM_TRIALS)
        results[snr] = {'ber': ber, 'bler': bler, 'throughput': thr}
        print(f"  BER = {ber:.6f}, BLER = {bler:.5f}, Throughput = {thr:.5f} бит/с/Гц")

    # Теоретические кривые
    snr_lin_vals = 10**(np.array(SNR_DB_LIST)/10)
    bler_theory = bler_theoretical(num_re * tx.bits_per_sym, snr_lin_vals, MODULATION)
    shannon = shannon_capacity(snr_lin_vals)
    spectral_eff_max = (tx.bits_per_sym * num_re) / (tx.fft_size + tx.cp_len)

    # BLER vs SNR
    plt.figure(figsize=(8,6))
    plt.semilogy(SNR_DB_LIST, [results[s]['bler'] for s in SNR_DB_LIST], 'bo-', label='Симуляция')
    plt.semilogy(SNR_DB_LIST, bler_theory, 'r--', label='Теория (приближённая)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('BLER')
    plt.title(f'{MODULATION} BLER для {BAND} BW={BW_MHZ}МГц SCS={SCS_KHZ}кГц')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("BLER_vs_SNR.png", dpi=150)

    # Throughput vs SNR
    plt.figure(figsize=(8,6))
    thr_sim = [results[s]['throughput'] for s in SNR_DB_LIST]
    plt.plot(SNR_DB_LIST, thr_sim, 'bo-', label=f'{MODULATION} с CP')
    plt.plot(SNR_DB_LIST, shannon, 'r--', label='Ёмкость Шеннона')
    plt.axhline(y=spectral_eff_max, color='gray', linestyle=':', label=f'Макс. SE = {spectral_eff_max:.3f}')
    plt.xlabel('SNR, дБ')
    plt.ylabel('Throughput, бит/с/Гц')
    plt.title(f'Спектральная эффективность, {MODULATION}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Throughput_vs_SNR.png", dpi=150)

    # Сохранение в папку
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"NTN_{MODULATION}_{BAND}_{BW_MHZ}MHz_{date_str}"
    os.makedirs(save_dir, exist_ok=True)
    os.rename("BLER_vs_SNR.png", os.path.join(save_dir, "BLER_vs_SNR.png"))
    os.rename("Throughput_vs_SNR.png", os.path.join(save_dir, "Throughput_vs_SNR.png"))

    # .mat данные
    mat_data = {
        'config': {'band': BAND, 'bw_mhz': BW_MHZ, 'scs_khz': SCS_KHZ,
                   'modulation': MODULATION, 'num_re': num_re, 'fft_size': fft_size,
                   'cp_len': tx.cp_len, 'max_spectral_eff': spectral_eff_max},
        'snr_db': SNR_DB_LIST,
        'bler_sim': np.array([results[s]['bler'] for s in SNR_DB_LIST]),
        'bler_theory': bler_theory,
        'throughput_sim': np.array(thr_sim),
        'shannon_capacity': shannon
    }
    savemat(os.path.join(save_dir, 'simulation_data.mat'), mat_data)

    print(f"\nРезультаты сохранены в папку: {save_dir}")
    print("Файлы: BLER_vs_SNR.png, Throughput_vs_SNR.png, simulation_data.mat")