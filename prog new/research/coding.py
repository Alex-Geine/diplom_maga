import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

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


# ------------------------------------------------------------
#  Моделирование с перемежением и адаптивной остановкой
# ------------------------------------------------------------
def bpsk_modulate(bits):
    return 1 - 2*bits.astype(float)

def bpsk_demodulate_llr(rx, noise_var):
    return 2 * rx / noise_var

def simulate_with_interleaver(encoder, decoder, interleaver, snr_db,
                              frame_len=500, min_errors=150, max_bits=int(5e6)):
    """
    snr_db: отношение Eb/N0 в dB (для BPSK c R=1/2)
    """
    snr_lin = 10**(snr_db/10)
    code_rate = 0.5
    es_n0 = snr_lin * code_rate
    noise_var = 1.0 / (2.0 * es_n0)   # BPSK, вещественный шум

    total_bits = 0
    total_errors = 0

    max_num_trials = 1e4
    trials = 0

    while (total_errors < min_errors and total_bits < max_bits) or trials < max_num_trials:
        # 1. Информационные биты
        info_bits = np.random.randint(0, 2, frame_len)

        # 2. Кодирование
        coded_bits = encoder.encode(info_bits)   # длина = 2*frame_len + 2*memory

        # 3. Перемежение (кодированные биты)
        interleaved_bits, orig_len, pad_len = interleaver.interleave(coded_bits)

        # 4. Модуляция
        tx_signal = bpsk_modulate(interleaved_bits)

        # 5. AWGN канал
        noise = np.sqrt(noise_var) * np.random.randn(len(tx_signal))
        rx_signal = tx_signal + noise

        # 6. LLR демодуляция
        llr = bpsk_demodulate_llr(rx_signal, noise_var)

        # 7. Деперемежение LLR (соответствующее перемежению битов)
        deinterleaved_llr = interleaver.deinterleave(llr, orig_len, pad_len)

        # 8. Декодирование Витерби
        decoded_bits = decoder.decode(deinterleaved_llr, block_len=frame_len)

        # 9. Подсчёт ошибок
        errors = np.sum(info_bits != decoded_bits)
        total_errors += errors
        total_bits += frame_len
        trials+= 1

    ber = total_errors / total_bits if total_bits > 0 else 0.0
    return ber


def main():
    snr_points_db = np.arange(0, 7, 1)   # 0 .. 6 dB
    frame_len = 500
    block_size = 1000   # размер блока перемежения (должен быть больше длины кадра?)

    encoder = ConvEncoder()
    decoder = ViterbiDecoder()
    # Создаём перемежитель с фиксированным seed для воспроизводимости
    interleaver = Interleaver(block_size=block_size, seed=123)

    ber_results = []
    print("SNR(dB)   BER")
    for snr in snr_points_db:
        ber = simulate_with_interleaver(encoder, decoder, interleaver,
                                        snr, frame_len=frame_len,
                                        min_errors=150, max_bits=int(5e6))
        ber_results.append(ber)
        print(f"{snr:4.1f}      {ber:.2e}")

    # Построение графика
    plt.figure(figsize=(8,5))
    plt.semilogy(snr_points_db, ber_results, 'o-', label='BER, soft Viterbi + interleaver')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Convolutional code (rate 1/2) with block interleaver, AWGN')
    plt.legend()
    plt.ylim([1e-5, 1])
    plt.show()

if __name__ == "__main__":
    main()