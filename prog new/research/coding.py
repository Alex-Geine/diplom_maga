import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

import numpy as np

# ---------- SoftDemapper (исправлен для BPSK) ----------
class SoftDemapper:
    def __init__(self, modulation):
        self.modulation = modulation
        if modulation == 'BPSK':
            self.bits_per_symbol = 1
            self.constellation = np.array([1, -1])
            self.bits = np.array([[0], [1]])
        elif modulation == 'QPSK':
            self.bits_per_symbol = 2
            self.constellation = np.array([1+1j, 1-1j, -1-1j, -1+1j]) / np.sqrt(2)
            self.bits = np.array([[0,0],[0,1],[1,1],[1,0]])
        elif modulation == '16QAM':
            self.bits_per_symbol = 4
            r = np.array([-3,-1,1,3])
            self.constellation = np.array([x+1j*y for x in r for y in r]) / np.sqrt(10)
            self.bits = self._gen_gray(4)
        # ... остальные модуляции добавьте по необходимости

    def _gen_gray(self, bps):
        # заглушка – для 16,64,256 используйте свой словарь, либо верните как есть
        raise NotImplementedError

    def demap(self, rx_symbols, noise_variance):
        rx_symbols = np.asarray(rx_symbols).flatten()
        if self.modulation == 'BPSK':
            # Простая и правильная формула для BPSK
            return 2 * np.real(rx_symbols) / noise_variance
        # Для остальных модуляций – универсальный max-log
        if not np.iscomplexobj(rx_symbols):
            rx_symbols = rx_symbols.astype(complex)
        num_symbols = len(rx_symbols)
        num_bits = num_symbols * self.bits_per_symbol
        llr = np.zeros(num_bits)
        for i, y in enumerate(rx_symbols):
            dist = np.abs(y - self.constellation)**2
            for bit in range(self.bits_per_symbol):
                idx0 = np.where(self.bits[:, bit] == 0)[0]
                idx1 = np.where(self.bits[:, bit] == 1)[0]
                d0 = np.min(dist[idx0]) if idx0.size else np.inf
                d1 = np.min(dist[idx1]) if idx1.size else np.inf
                llr[i*self.bits_per_symbol + bit] = (d0 - d1) / (2 * noise_variance)
        return llr

# ------------------------------------------------------------
#  Блочный перемежитель (случайная перестановка)
# ------------------------------------------------------------
class Interleaver:
    def __init__(self, block_size, seed=42):
        """
        block_size: размер блока для перемежения (должен быть кратен длине кадра)
        seed: для воспроизводимости
        """
        self.block_size = block_size
        self.rng = RandomState(seed)
        # Генерируем случайную перестановку индексов
        self.interleaver_pattern = self.rng.permutation(block_size)
        # Обратная перестановка для деперемежения
        self.deinterleaver_pattern = np.argsort(self.interleaver_pattern)

    def interleave(self, bits):
        """Перемежение битового массива (должен быть кратен block_size)"""
        bits = np.asarray(bits)
        # Дополняем до целого числа блоков, если нужно
        orig_len = len(bits)
        pad_len = (self.block_size - orig_len % self.block_size) % self.block_size
        if pad_len > 0:
            bits = np.pad(bits, (0, pad_len), constant_values=0)
        bits_reshaped = bits.reshape(-1, self.block_size)
        interleaved = bits_reshaped[:, self.interleaver_pattern].ravel()
        return interleaved, orig_len, pad_len

    def deinterleave(self, bits, orig_len, pad_len):
        """Деперемежение с удалением дополненных битов"""
        bits = np.asarray(bits)
        bits_reshaped = bits.reshape(-1, self.block_size)
        deinterleaved = bits_reshaped[:, self.deinterleaver_pattern].ravel()
        if pad_len > 0:
            deinterleaved = deinterleaved[:orig_len]
        return deinterleaved


# ------------------------------------------------------------
#  Свёрточный кодер (2,1,7) – без изменений
# ------------------------------------------------------------
class ConvEncoder:
    def __init__(self, poly1=0o133, poly2=0o171, memory=6):
        self.poly1 = poly1
        self.poly2 = poly2
        self.memory = memory

    def encode(self, bits):
        bits = np.asarray(bits, dtype=int)
        reg = 0
        out = []
        for b in bits:
            reg = ((reg << 1) & 0x7F) | b
            out1 = bin(reg & self.poly1).count('1') & 1
            out2 = bin(reg & self.poly2).count('1') & 1
            out.extend([out1, out2])
        # Tail (memory нулей)
        for _ in range(self.memory):
            reg = (reg << 1) & 0x7F
            out1 = bin(reg & self.poly1).count('1') & 1
            out2 = bin(reg & self.poly2).count('1') & 1
            out.extend([out1, out2])
        return np.array(out, dtype=int)


# ------------------------------------------------------------
#  Декодер Витерби (мягкие решения) – без изменений
# ------------------------------------------------------------
class ViterbiDecoder:
    def __init__(self, poly1=0o133, poly2=0o171, memory=6):
        self.poly1 = poly1
        self.poly2 = poly2
        self.memory = memory
        self.num_states = 1 << memory
        # предвычисление переходов
        self.transition = []
        for state in range(self.num_states):
            row = []
            for inp in (0, 1):
                reg = ((state << 1) & 0x7F) | inp
                out1 = bin(reg & self.poly1).count('1') & 1
                out2 = bin(reg & self.poly2).count('1') & 1
                next_state = reg & (self.num_states - 1)
                row.append((next_state, out1, out2))
            self.transition.append(row)

    def decode(self, llr_seq, block_len):
        N = len(llr_seq) // 2
        num_states = self.num_states
        path_metrics = np.full(num_states, np.inf)
        path_metrics[0] = 0.0
        traceback = []

        for step in range(N):
            llr1 = llr_seq[2*step]
            llr2 = llr_seq[2*step+1]
            new_metrics = np.full(num_states, np.inf)
            survivors = [None] * num_states
            for s in range(num_states):
                for inp in (0, 1):
                    ns, o1, o2 = self.transition[s][inp]
                    metric = - (llr1*(1-2*o1) + llr2*(1-2*o2))
                    cand = path_metrics[s] + metric
                    if cand < new_metrics[ns]:
                        new_metrics[ns] = cand
                        survivors[ns] = (s, inp)
            path_metrics = new_metrics
            traceback.append(survivors)

        best_state = np.argmin(path_metrics)
        decoded = []
        cur_state = best_state
        for step in range(N-1, -1, -1):
            prev_state, inp_bit = traceback[step][cur_state]
            decoded.append(inp_bit)
            cur_state = prev_state
        decoded.reverse()
        return np.array(decoded[:block_len], dtype=int)

# ---------- Симуляция с комплексным шумом и внешним демодулятором ----------
def simulate_with_interleaver(encoder, decoder, interleaver, demapper, snr_db,
                              frame_len=500, min_errors=150, max_bits=int(5e6)):
    # SNR -> дисперсия шума на компоненту (I/Q)
    snr_lin = 10**(snr_db/10)
    code_rate = 0.5
    es_n0 = snr_lin * code_rate
    noise_var = 1.0 / (2.0 * es_n0)   # дисперсия для реальной и мнимой части

    total_bits = 0
    total_errors = 0

    trials = 0
    max_trials = 1e4

    while total_errors < min_errors and total_bits < max_bits and trials < max_trials:
        info = np.random.randint(0, 2, frame_len)
        coded = encoder.encode(info)
        inter, orig_len, pad = interleaver.interleave(coded)

        # Модуляция BPSK -> +1/-1, комплексные символы (мнимая часть 0)
        tx = 1 - 2 * inter.astype(float)   # вещественные
        tx = tx.astype(complex)            # теперь комплексные

        # Комплексный AWGN
        noise = np.sqrt(noise_var) * (np.random.randn(len(tx)) + 1j*np.random.randn(len(tx)))
        rx = tx + noise

        # Мягкая демодуляция
        llr = demapper.demap(rx, noise_var)

        # Деперемежение LLR
        deinter_llr = interleaver.deinterleave(llr, orig_len, pad)

        # Декодирование
        decoded = decoder.decode(deinter_llr, block_len=frame_len)

        errors = np.sum(info != decoded)
        total_errors += errors
        total_bits += frame_len
        trials += 1

    return total_errors / total_bits if total_bits else 0.0

def main():
    snr_db = np.arange(0, 7, 1)
    frame_len = 500
    block_size = 1000

    encoder = ConvEncoder()
    decoder = ViterbiDecoder()
    interleaver = Interleaver(block_size, seed=123)
    demapper = SoftDemapper('BPSK')   # создаём один раз

    ber_list = []
    for snr in snr_db:
        ber = simulate_with_interleaver(encoder, decoder, interleaver, demapper,
                                        snr, frame_len, min_errors=150, max_bits=int(5e6))
        ber_list.append(ber)
        print(f"{snr:3.1f} dB   BER = {ber:.2e}")

    plt.semilogy(snr_db, ber_list, 'o-')
    plt.grid(True)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Convolutional code + interleaver, BPSK, AWGN')
    plt.ylim([1e-6, 1])
    plt.show()

if __name__ == '__main__':
    main()