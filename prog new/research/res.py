import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
import os
import datetime

# ========================= 1. Параметры 5G NTN =========================
# Таблица CP
cp_table = {
    512:  {'normal': 36, 'extended': 40},
    1024: {'normal': 72, 'extended': 80},
    2048: {'normal': 144, 'extended': 160},
    4096: {'normal': 288, 'extended': 320}
}
def get_cp_length(fft_size, cp_type='normal'):
    return cp_table[fft_size][cp_type]

# Таблица RB
ls_rb_table = {5: {15:25,30:11}, 10:{15:52,30:24}, 15:{15:79,30:38}, 20:{15:106,30:51}}
ka_rb_table = {50:{60:66,120:32}, 100:{60:132,120:66}, 200:{60:264,120:132}, 400:{120:264}}
def get_rb(band, bw_mhz, scs_khz):
    if band == 'L_S': return ls_rb_table.get(bw_mhz, {}).get(scs_khz)
    elif band == 'Ka': return ka_rb_table.get(bw_mhz, {}).get(scs_khz)
    return None

def get_fft_size_from_re(num_re):
    n = 64
    while n <= num_re: n *= 2
    return n

# ========================= 2. Модуляция и демодуляция =========================
def modulate(bits, mod_type):
    bits = np.asarray(bits)
    if mod_type == 'BPSK':
        return 2*bits - 1
    elif mod_type == 'QPSK':
        even, odd = bits[0::2], bits[1::2]
        return (1-2*even + 1j*(1-2*odd)) / np.sqrt(2)
    elif mod_type == '16QAM':
        b = bits.reshape(-1,4)
        I = (1-2*b[:,0]) * (2 - (1-2*b[:,1]))
        Q = (1-2*b[:,2]) * (2 - (1-2*b[:,3]))
        return (I + 1j*Q) / np.sqrt(10)
    elif mod_type == '64QAM':
        b = bits.reshape(-1,6)
        I = (1-2*b[:,0]) * (4 - (1-2*b[:,1])*(2 - (1-2*b[:,2])))
        Q = (1-2*b[:,3]) * (4 - (1-2*b[:,4])*(2 - (1-2*b[:,5])))
        return (I + 1j*Q) / np.sqrt(42)
    elif mod_type == '256QAM':
        b = bits.reshape(-1,8)
        I = (1-2*b[:,0]) * (8 - (1-2*b[:,1])*(4 - (1-2*b[:,2])*(2 - (1-2*b[:,3]))))
        Q = (1-2*b[:,4]) * (8 - (1-2*b[:,5])*(4 - (1-2*b[:,6])*(2 - (1-2*b[:,7]))))
        return (I + 1j*Q) / np.sqrt(170)
    else:
        raise ValueError(f"Unknown modulation: {mod_type}")

def demodulate(symbols, mod_type):
    if mod_type == 'BPSK':
        return (np.real(symbols) > 0).astype(int)
    elif mod_type == 'QPSK':
        s = symbols * np.sqrt(2)
        bits = np.empty(len(symbols)*2, dtype=int)
        bits[0::2] = (np.real(s) < 0).astype(int)
        bits[1::2] = (np.imag(s) < 0).astype(int)
        return bits
    elif mod_type == '16QAM':
        s = symbols * np.sqrt(10)
        I, Q = np.real(s), np.imag(s)
        bits = np.empty(len(symbols)*4, dtype=int)
        for i, (x,y) in enumerate(zip(I,Q)):
            # I: пороги -2,0,2 -> 2 бита
            if x < -2: b0,b1 = 1,1
            elif x < 0: b0,b1 = 1,0
            elif x < 2: b0,b1 = 0,1
            else: b0,b1 = 0,0
            if y < -2: b2,b3 = 1,1
            elif y < 0: b2,b3 = 1,0
            elif y < 2: b2,b3 = 0,1
            else: b2,b3 = 0,0
            bits[4*i:4*i+4] = [b0,b1,b2,b3]
        return bits
    elif mod_type == '64QAM':
        s = symbols * np.sqrt(42)
        I, Q = np.real(s), np.imag(s)
        bits = np.empty(len(symbols)*6, dtype=int)
        for i, (x,y) in enumerate(zip(I,Q)):
            # Пороги -6,-4,-2,0,2,4,6
            if x < -6: b0,b1,b2 = 1,1,1
            elif x < -4: b0,b1,b2 = 1,1,0
            elif x < -2: b0,b1,b2 = 1,0,1
            elif x < 0: b0,b1,b2 = 1,0,0
            elif x < 2: b0,b1,b2 = 0,1,1
            elif x < 4: b0,b1,b2 = 0,1,0
            elif x < 6: b0,b1,b2 = 0,0,1
            else: b0,b1,b2 = 0,0,0
            if y < -6: b3,b4,b5 = 1,1,1
            elif y < -4: b3,b4,b5 = 1,1,0
            elif y < -2: b3,b4,b5 = 1,0,1
            elif y < 0: b3,b4,b5 = 1,0,0
            elif y < 2: b3,b4,b5 = 0,1,1
            elif y < 4: b3,b4,b5 = 0,1,0
            elif y < 6: b3,b4,b5 = 0,0,1
            else: b3,b4,b5 = 0,0,0
            bits[6*i:6*i+6] = [b0,b1,b2,b3,b4,b5]
        return bits
    elif mod_type == '256QAM':
        s = symbols * np.sqrt(170)
        I, Q = np.real(s), np.imag(s)
        bits = np.empty(len(symbols)*8, dtype=int)
        levels = np.arange(-15, 16, 2)  # -15,-13,...,15
        thresh = np.arange(-14, 15, 2)
        def slice256(x):
            idx = np.digitize(x, thresh, right=False)
            # 4-бит non-Gray для простоты
            return [(idx>>3)&1, (idx>>2)&1, (idx>>1)&1, idx&1]
        for i, (x,y) in enumerate(zip(I,Q)):
            ibits = slice256(x)
            qbits = slice256(y)
            bits[8*i:8*i+8] = ibits + qbits
        return bits
    else:
        raise ValueError(f"Unknown demodulation: {mod_type}")

def bits_per_symbol(mod_type):
    return {'BPSK':1, 'QPSK':2, '16QAM':4, '64QAM':6, '256QAM':8}[mod_type]

# ========================= 3. OFDM передатчик/приёмник =========================
class OFDMTx:
    def __init__(self, num_re, fft_size, cp_type='normal', mod_type='QPSK'):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2
        self.mod_type = mod_type
        self.bps = bits_per_symbol(mod_type)

    def transmit(self, bits):
        # bits: массив длины num_re * bps
        symbols = modulate(bits, self.mod_type)
        freq = np.zeros(self.fft_size, dtype=complex)
        freq[self.offset:self.offset+self.num_re] = symbols
        time_signal = np.fft.ifft(freq, norm='ortho')
        # добавить CP
        return np.concatenate([time_signal[-self.cp_len:], time_signal])

class OFDMRx:
    def __init__(self, num_re, fft_size, cp_type='normal', mod_type='QPSK'):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2
        self.mod_type = mod_type
        self.bps = bits_per_symbol(mod_type)

    def receive(self, rx_signal, H_freq, N0):
        # удалить CP
        signal = rx_signal[self.cp_len:self.cp_len+self.fft_size]
        Y = np.fft.fft(signal, norm='ortho')
        Y_re = Y[self.offset:self.offset+self.num_re]
        H_re = H_freq[self.offset:self.offset+self.num_re]
        # MMSE эквалайзер
        H_conj = np.conj(H_re)
        X_hat = H_conj / (np.abs(H_re)**2 + N0) * Y_re
        bits = demodulate(X_hat, self.mod_type)
        return bits

# ========================= 4. TDL канал для 5G NTN =========================
# Профили TDL-A, B, C из 3GPP TR 38.901 (задержки в нс, мощности в дБ)
TDL_PROFILES = {
    'A': (  # TDL-A (нелинейный профиль, NLOS)
        [0, 30, 70, 90, 110, 190, 410, 530, 750, 1070, 1090, 1290],
        [-13.4, -0.0, -9.2, -7.5, -4.7, -11.0, -4.7, -16.6, -11.8, -13.4, -19.4, -24.8]
    ),
    'B': (  # TDL-B (средний NLOS)
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        [-7.8, -6.2, -7.2, -8.6, -7.5, -10.0, -8.7, -11.0, -11.2, -12.8, -13.4, -14.5, -15.2, -16.9, -17.2, -18.0, -19.4, -20.7, -21.3, -26.4]
    ),
    'C': (  # TDL-C (LOS, с сильным первым лучом)
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        [-3.5, -6.8, -8.2, -9.2, -10.2, -11.1, -12.2, -13.2, -14.1, -15.1, -17.6, -18.7, -19.7, -20.7, -21.7, -22.7, -23.7, -24.7, -25.7, -26.7]
    )
}

class TDLChannel:
    def __init__(self, profile_name, fft_size, scs_khz, d_km, fc_ghz, shadowing_std_db, cp_len):
        """
        profile_name: 'A', 'B', 'C'
        fft_size: размер БПФ
        scs_khz: шаг поднесущих в кГц
        d_km: расстояние (высота орбиты) в км
        fc_ghz: несущая частота в ГГц
        shadowing_std_db: стандартное отклонение теневых потерь (дБ)
        """
        self.fft_size = fft_size
        self.scs_hz = scs_khz * 1e3
        self.fs = fft_size * self.scs_hz   # частота дискретизации
        self.Ts = 1 / self.fs              # период дискретизации в секундах
        self.d_m = d_km * 1e3
        self.fc = fc_ghz * 1e9
        self.c = 3e8
        self.lambda_c = self.c / self.fc
        self.shadowing_std_db = shadowing_std_db
        # Потери свободного пространства (freespace path loss) для заданного d
        self.Pd_lin = (4 * np.pi * self.d_m / self.lambda_c) ** 2   # линейные потери (мощность)
        self.Pd_db = 10 * np.log10(self.Pd_lin)   # дБ
        self.cp_len = cp_len   # сохраняем длину CP

        # Загружаем профиль TDL
        delays_ns, powers_db = TDL_PROFILES[profile_name]
        # Преобразуем задержки в отсчёты (округляем вниз/вверх? используем округление до целого)
        self.delays_samples = np.round(np.array(delays_ns) * 1e-9 / self.Ts ).astype(int)
        self.path_gains_lin = 10 ** (np.array(powers_db) / 10.0)   # линейные коэффициенты без учёта Pd, Ps
        # Максимальная задержка (в отсчётах)
        self.max_delay = np.max(self.delays_samples)
        # Длина импульсной характеристики (с учётом возможной дополнительной задержки)
        self.ir_len = self.max_delay + 1

    def get_path_loss_factor(self):
        """
        Возвращает общий коэффициент ослабления sqrt(P), где P = 10^(-(Pd+Ps)/10).
        Ps - случайная логнормальная величина (shadowing).
        """
        Ps_db = np.random.normal(0, self.shadowing_std_db)
        total_loss_db = self.Pd_db + Ps_db
        P_lin = 10 ** (-total_loss_db / 10.0)
        return np.sqrt(P_lin)   # потому что h(t) умножается на sqrt(P)

    def get_impulse_response(self):
        """Возвращает комплексную импульсную характеристику канала (в отсчётах) с учётом потерь."""
        path_loss_factor = self.get_path_loss_factor()
        h = np.zeros(self.ir_len, dtype=complex)

        for gain_lin, delay in zip(self.path_gains_lin, self.delays_samples):
            amp = np.sqrt(gain_lin) * path_loss_factor
            phase = 2 * np.pi * np.random.rand()
            h[delay] += amp * np.exp(1j * phase)
        return h

    def apply(self, tx_signal, snr_lin):
        # 1. Извлекаем полезный OFDM-символ (удаляем CP)
        signal_no_cp = tx_signal[self.cp_len:self.cp_len + self.fft_size]

        # 2. БПФ переданного символа
        X_freq = np.fft.fft(signal_no_cp, norm='ortho')

        # 3. Импульсная характеристика канала
        h = self.get_impulse_response()
        h_full = np.zeros(self.fft_size, dtype=complex)
        h_full[:len(h)] = h
        H_freq = np.fft.fft(h_full, norm='ortho')

        # 4. Прохождение через канал (умножение спектров)
        Y_freq = X_freq * H_freq

        # 5. Обратное БПФ для получения временного сигнала после канала (без шума)
        y_time_no_noise = np.fft.ifft(Y_freq, norm='ortho')

        # 6. Восстанавливаем CP (копируем последние cp_len отсчётов)
        cp = y_time_no_noise[-self.cp_len:]
        rx_signal_with_cp = np.concatenate([cp, y_time_no_noise])

        # 7. Мощность принятого сигнала и добавление шума
        P_rx = np.mean(np.abs(rx_signal_with_cp)**2)
        N0 = P_rx / snr_lin
        noise = np.sqrt(N0/2) * (np.random.randn(*rx_signal_with_cp.shape) + 1j*np.random.randn(*rx_signal_with_cp.shape))
        rx_signal_with_cp += noise

        return rx_signal_with_cp, H_freq, N0

# ========================= 5. Теоретические кривые =========================
def ber_awgn_qam(snr_lin, mod_type):
    M = 2**bits_per_symbol(mod_type)
    if M == 2:
        return 0.5 * erfc(np.sqrt(snr_lin))
    elif M == 4:
        return 0.5 * erfc(np.sqrt(snr_lin/2))
    else:
        k = bits_per_symbol(mod_type)
        return (4/k)*(1-1/np.sqrt(M))*0.5*erfc(np.sqrt(3*k*snr_lin/(2*(M-1))))

def bler_theoretical(num_bits, snr_lin, mod_type):
    ber = ber_awgn_qam(snr_lin, mod_type)
    return 1 - (1 - ber)**num_bits

def shannon_capacity(snr_lin):
    return np.log2(1 + snr_lin)

# ========================= 6. Симуляция для одного SNR =========================
def simulate_snr(snr_db, tx, rx, tdl_channel, num_trials):
    snr_lin = 10**(snr_db/10.0)
    total_bit_errors = 0
    block_errors = 0
    bits_per_block = tx.num_re * tx.bps

    for _ in range(num_trials):
        bits_tx = np.random.randint(0, 2, bits_per_block)
        tx_signal = tx.transmit(bits_tx)
        rx_signal, H_freq, N0 = tdl_channel.apply(tx_signal, snr_lin)
        bits_rx = rx.receive(rx_signal, H_freq, N0)
        errors = np.sum(bits_tx != bits_rx)
        total_bit_errors += errors
        if errors > 0:
            block_errors += 1

    ber = total_bit_errors / (bits_per_block * num_trials)
    bler = block_errors / num_trials
    spectral_eff_max = (tx.bps * tx.num_re) / (tx.fft_size + tx.cp_len)
    throughput = (1 - ber) * spectral_eff_max
    return ber, bler, throughput

# ========================= 7. Основной скрипт =========================
if __name__ == "__main__":
    # Параметры системы
    BAND = 'L_S'               # 'L_S' или 'Ka'
    BW_MHZ = 10                # полоса, МГц
    SCS_KHZ = 30               # шаг поднесущих, кГц
    MODULATION = 'QPSK'        # 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'

    # Параметры орбиты и канала
    ORBIT_HEIGHT_KM = 600      # высота (расстояние) в км
    CARRIER_FREQ_GHZ = 2.0     # несущая частота (ГГц) для L/S диапазона, для Ka ~20
    SHADOWING_STD_DB = 3.0     # стандартное отклонение теневых потерь (дБ)
    TDL_PROFILE = 'C'          # 'A', 'B', 'C'

    # Параметры симуляции
    SNR_DB_LIST = np.arange(0, 21, 2)   # 0..20 дБ шаг 2
    NUM_TRIALS = 1000           # число OFDM-символов на SNR

    # Расчёт конфигурации
    rb = get_rb(BAND, BW_MHZ, SCS_KHZ)
    if rb is None:
        print("Ошибка: Неверная комбинация диапазон/полоса/SCS")
        exit(1)
    num_re = rb * 12
    fft_size = get_fft_size_from_re(num_re)
    cp_type = 'normal'

    # Создаём передатчик и приёмник (статичные)
    tx = OFDMTx(num_re, fft_size, cp_type, MODULATION)
    rx = OFDMRx(num_re, fft_size, cp_type, MODULATION)

    # Создаём TDL канал (общий для всех SNR, но внутри будет генерироваться случайный shadowing)
    tdl = TDLChannel(TDL_PROFILE, fft_size, SCS_KHZ,
                     ORBIT_HEIGHT_KM, CARRIER_FREQ_GHZ, SHADOWING_STD_DB, tx.cp_len)

    print("===== Конфигурация =====")
    print(f"Диапазон: {BAND}, BW={BW_MHZ} МГц, SCS={SCS_KHZ} кГц -> RB={rb}, RE={num_re}")
    print(f"FFT size = {fft_size}, CP length = {tx.cp_len} отсч.")
    print(f"Модуляция: {MODULATION} ({tx.bps} бит/символ)")
    print(f"Орбита: {ORBIT_HEIGHT_KM} км, fc={CARRIER_FREQ_GHZ} ГГц, Shadowing σ={SHADOWING_STD_DB} дБ")
    print(f"Профиль TDL: {TDL_PROFILE}")
    print("=========================")

    results = {}
    for snr in SNR_DB_LIST:
        print(f"Симуляция SNR = {snr} дБ ...")
        ber, bler, thr = simulate_snr(snr, tx, rx, tdl, NUM_TRIALS)
        results[snr] = {'ber': ber, 'bler': bler, 'throughput': thr}
        print(f"  BER = {ber:.6f}, BLER = {bler:.5f}, Throughput = {thr:.5f} бит/с/Гц")

    # Теоретические кривые (для справки, но они не учитывают многолучевость и эквалайзер)
    snr_lin_vals = 10**(np.array(SNR_DB_LIST)/10)
    bits_per_block = num_re * tx.bps
    bler_theory = bler_theoretical(bits_per_block, snr_lin_vals, MODULATION)
    shannon = shannon_capacity(snr_lin_vals)
    max_se = (tx.bps * num_re) / (fft_size + tx.cp_len)

    # Построение BLER vs SNR
    plt.figure(figsize=(8,6))
    plt.semilogy(SNR_DB_LIST, [results[s]['bler'] for s in SNR_DB_LIST], 'bo-', label='Симуляция (TDL+MMSE)')
    plt.semilogy(SNR_DB_LIST, bler_theory, 'r--', label='Теория (AWGN, без многолучевости)')
    plt.xlabel('SNR, дБ')
    plt.ylabel('BLER')
    plt.title(f'BLER в {MODULATION}, NTN TDL-{TDL_PROFILE}, орбита {ORBIT_HEIGHT_KM} км')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    bler_fig = plt.gcf()

    # Throughput vs SNR + Шеннон
    plt.figure(figsize=(8,6))
    thr_sim = [results[s]['throughput'] for s in SNR_DB_LIST]
    plt.plot(SNR_DB_LIST, thr_sim, 'bo-', label=f'{MODULATION} (TDL+MMSE)')
    plt.plot(SNR_DB_LIST, shannon, 'r--', label='Ёмкость Шеннона')
    plt.axhline(y=max_se, color='gray', linestyle=':', label=f'Макс. SE = {max_se:.3f}')
    plt.xlabel('SNR, дБ')
    plt.ylabel('Throughput, бит/с/Гц')
    plt.title(f'Спектральная эффективность, TDL-{TDL_PROFILE}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    thr_fig = plt.gcf()

    # Сохранение
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"NTN_{MODULATION}_{TDL_PROFILE}_H{ORBIT_HEIGHT_KM}km_{date_str}"
    os.makedirs(save_dir, exist_ok=True)
    bler_fig.savefig(os.path.join(save_dir, "BLER_vs_SNR.png"), dpi=150)
    thr_fig.savefig(os.path.join(save_dir, "Throughput_vs_SNR.png"), dpi=150)
    plt.close('all')

    # Сохраняем .mat c данными
    mat_data = {
        'config': {
            'band': BAND, 'bw_mhz': BW_MHZ, 'scs_khz': SCS_KHZ,
            'modulation': MODULATION, 'num_re': num_re, 'fft_size': fft_size,
            'cp_len': tx.cp_len, 'max_spectral_eff': max_se,
            'orbit_km': ORBIT_HEIGHT_KM, 'carrier_ghz': CARRIER_FREQ_GHZ,
            'shadowing_std_db': SHADOWING_STD_DB, 'tdl_profile': TDL_PROFILE
        },
        'snr_db': SNR_DB_LIST,
        'bler_sim': np.array([results[s]['bler'] for s in SNR_DB_LIST]),
        'bler_theory_awgn': bler_theory,
        'throughput_sim': np.array(thr_sim),
        'shannon_capacity': shannon
    }
    savemat(os.path.join(save_dir, 'simulation_data.mat'), mat_data)

    print(f"\nРезультаты сохранены в папку: {save_dir}")
    print("Файлы: BLER_vs_SNR.png, Throughput_vs_SNR.png, simulation_data.mat")