import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ------------------------------------------------------------
#  Свёрточный кодер (2,1,7)
# ------------------------------------------------------------
class ConvEncoder:
    def __init__(self, poly1=0o133, poly2=0o171, memory=6):
        """
        poly1, poly2 - полиномы в восьмеричной форме
        memory - длина регистра сдвига (степень кода)
        """
        self.poly1 = poly1
        self.poly2 = poly2
        self.memory = memory
        self.state_mask = (1 << memory) - 1   # mask для младших memory бит

    def encode(self, bits):
        """
        Вход: bits - список/массив из 0/1
        Выход: numpy array кодированных битов (длина в 2 раза больше + tail)
        """
        bits = np.asarray(bits, dtype=int)
        reg = 0  # состояние кодера (7 бит, но используются младшие memory+1)
        out_bits = []

        for b in bits:
            reg = ((reg << 1) & 0x7F) | b   # сдвигаем и добавляем новый бит (7-битный регистр)
            # вычисляем два выходных бита как XOR битов, где полином имеет 1
            out1 = bin(reg & self.poly1).count('1') % 2
            out2 = bin(reg & self.poly2).count('1') % 2
            out_bits.extend([out1, out2])

        # Tail biting: добавляем memory нулевых битов для сброса регистра в 0
        for _ in range(self.memory):
            reg = (reg << 1) & 0x7F
            out1 = bin(reg & self.poly1).count('1') % 2
            out2 = bin(reg & self.poly2).count('1') % 2
            out_bits.extend([out1, out2])

        return np.array(out_bits, dtype=int)


# ------------------------------------------------------------
#  Декодер Витерби (мягкие решения, алгоритм с метриками Хемминга/Евклида)
# ------------------------------------------------------------
class ViterbiDecoder:
    def __init__(self, poly1=0o133, poly2=0o171, memory=6, decoding_type='soft'):
        """
        decoding_type: 'soft' - использует LLR, 'hard' - жёсткие биты
        """
        self.poly1 = poly1
        self.poly2 = poly2
        self.memory = memory
        self.num_states = 1 << memory   # 64 состояния
        self.decoding_type = decoding_type

        # Предвычисление выходных бит кодера для каждого состояния и входного бита
        # Структура: next_state, out1, out2 = transition[state][input_bit]
        self.transition = []
        for state in range(self.num_states):
            row = []
            for inp in (0, 1):
                reg = ((state << 1) & 0x7F) | inp
                out1 = bin(reg & self.poly1).count('1') % 2
                out2 = bin(reg & self.poly2).count('1') % 2
                next_state = reg & (self.num_states - 1)
                row.append((next_state, out1, out2))
            self.transition.append(row)

    def _branch_metric_hard(self, bits_in, out1, out2):
        """Хеммингова метрика для жёсткого декодирования"""
        return (bits_in[0] != out1) + (bits_in[1] != out2)

    def _branch_metric_soft(self, llr1, llr2, out1, out2):
        """
        Евклидова метрика для мягкого декодирования.
        Преобразуем LLR в символы BPSK: 0 -> +1, 1 -> -1.
        Расстояние (y - x)^2, где y - принятый символ (tanh(LLR/2)).
        """
        # Используем упрощённое выражение: метрика = LLR * (2*out - 1)
        # (эквивалентно максимизации правдоподобия для BPSK)
        # out = 0 -> символ +1; out = 1 -> символ -1
        sym = 1 - 2 * out1   # +1 для 0, -1 для 1
        metric1 = llr1 * sym
        sym = 1 - 2 * out2
        metric2 = llr2 * sym
        return -(metric1 + metric2)   # знак минус, чтобы искать минимум (или метрика расстояния)

    def decode(self, received, block_len=None):
        """
        received: массив LLR для мягкого режима, либо битов для жёсткого.
        block_len: длина исходных информационных битов (без учёта tail). Если None, то
                   предполагается, что received включает хвостовые биты.
        Возвращает: декодированные информационные биты.
        """
        if self.decoding_type == 'soft':
            # received - массив LLR двойной длины (info_bits*2 + tail*2)
            return self._viterbi_soft(received, block_len)
        else:
            return self._viterbi_hard(received, block_len)

    def _viterbi_soft(self, llr_seq, block_len=None):
        N = len(llr_seq) // 2   # количество шагов (пары выходных битов)
        if block_len is None:
            block_len = N - self.memory   # вычитаем tail

        # Инициализация: метрики путей (64 состояния)
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0.0
        # Храним предков и переданные биты
        traceback = []   # каждый элемент: массив из (prev_state, input_bit) для каждого состояния

        for step in range(N):
            # Берем пару LLR
            llr1 = llr_seq[2*step]
            llr2 = llr_seq[2*step+1]
            new_metrics = np.full(self.num_states, np.inf)
            survivors = [None] * self.num_states  # (prev_state, input_bit)

            for cur_state in range(self.num_states):
                for inp_bit in (0, 1):
                    next_state, out1, out2 = self.transition[cur_state][inp_bit]
                    metric = self._branch_metric_soft(llr1, llr2, out1, out2)
                    candidate = path_metrics[cur_state] + metric
                    if candidate < new_metrics[next_state]:
                        new_metrics[next_state] = candidate
                        survivors[next_state] = (cur_state, inp_bit)

            path_metrics = new_metrics
            traceback.append(survivors)

        # Выбираем лучшее конечное состояние (среди всех, но часто берут нулевое для tail)
        best_state = np.argmin(path_metrics)
        # Восстанавливаем биты в обратном порядке
        decoded_bits = []
        cur_state = best_state
        for step in range(N-1, -1, -1):
            _, inp_bit = traceback[step][cur_state]
            decoded_bits.append(inp_bit)
            cur_state = traceback[step][cur_state][0]
        decoded_bits.reverse()

        # Обрезаем tail-биты (первые block_len битов - информационные)
        return np.array(decoded_bits[:block_len], dtype=int)

    def _viterbi_hard(self, bit_seq, block_len=None):
        N = len(bit_seq) // 2
        if block_len is None:
            block_len = N - self.memory

        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0
        traceback = []

        for step in range(N):
            b1 = bit_seq[2*step]
            b2 = bit_seq[2*step+1]
            new_metrics = np.full(self.num_states, np.inf)
            survivors = [None] * self.num_states

            for cur_state in range(self.num_states):
                for inp_bit in (0, 1):
                    next_state, out1, out2 = self.transition[cur_state][inp_bit]
                    metric = self._branch_metric_hard((b1, b2), out1, out2)
                    candidate = path_metrics[cur_state] + metric
                    if candidate < new_metrics[next_state]:
                        new_metrics[next_state] = candidate
                        survivors[next_state] = (cur_state, inp_bit)

            path_metrics = new_metrics
            traceback.append(survivors)

        best_state = np.argmin(path_metrics)
        decoded_bits = []
        cur_state = best_state
        for step in range(N-1, -1, -1):
            _, inp_bit = traceback[step][cur_state]
            decoded_bits.append(inp_bit)
            cur_state = traceback[step][cur_state][0]
        decoded_bits.reverse()
        return np.array(decoded_bits[:block_len], dtype=int)


# ------------------------------------------------------------
#  Моделирование
# ------------------------------------------------------------
def generate_random_bits(N):
    return np.random.randint(0, 2, N)

def bpsk_modulate(bits):
    # 0 -> +1, 1 -> -1
    return 1 - 2 * bits.astype(float)

def bpsk_demodulate_llr(rx, noise_var):
    # LLR для BPSK: L = 2*y / sigma^2
    return 2 * rx / noise_var

def calculate_ber(orig, decoded):
    if len(orig) != len(decoded):
        # Обрезаем до минимальной длины
        min_len = min(len(orig), len(decoded))
        orig = orig[:min_len]
        decoded = decoded[:min_len]
    errors = np.sum(orig != decoded)
    return errors / len(orig)

def simulate_awgn(encoder, decoder, snr_db_list, num_frames=100, frame_len=1000):
    """
    Запускает симуляцию для списка SNR в dB.
    Возвращает массивы BER и BLER для каждого SNR.
    """
    ber_list = []
    bler_list = []

    for snr_db in snr_db_list:
        print("snr_db: ", snr_db)
        snr_lin = 10**(snr_db/10.0)
        # Для BPSK: Eb/N0 = SNR, но учтём скорость кода 1/2: Es/N0 = Eb/N0 * rate
        # В модели BPSK символьная энергия = 1, поэтому variance шума = 1/(2 * EsN0)
        # EsN0 = (EbN0) * code_rate. Здесь EbN0 = snr_lin (так как битовая энергия = 1)
        code_rate = 0.5
        es_n0 = snr_lin * code_rate
        noise_var = 1.0 / (2.0 * es_n0)   # комплексный шум, но для вещественного BPSK variance добавляем

        total_bits = 0
        total_errors = 0
        total_blocks = 0
        block_errors = 0

        for i in range(num_frames):
            if (not (i % 100)):
                print(i, "/", num_frames)
            # Генерация информационных битов
            info_bits = generate_random_bits(frame_len)
            # Кодирование
            coded_bits = encoder.encode(info_bits)
            # Модуляция BPSK
            tx_signal = bpsk_modulate(coded_bits)
            # Добавление AWGN
            noise = np.sqrt(noise_var) * np.random.randn(len(tx_signal))
            rx_signal = tx_signal + noise
            # LLR демодуляция
            llr = bpsk_demodulate_llr(rx_signal, noise_var)
            # Декодирование (мягкие решения)
            decoded_bits = decoder.decode(llr, block_len=frame_len)
            # Сравнение
            errors = np.sum(info_bits != decoded_bits)
            total_errors += errors
            total_bits += len(info_bits)
            if errors > 0:
                block_errors += 1
            total_blocks += 1

        ber = total_errors / total_bits if total_bits > 0 else 0
        bler = block_errors / total_blocks if total_blocks > 0 else 0
        ber_list.append(ber)
        bler_list.append(bler)
        print(f"SNR: {snr_db} dB, BER: {ber:.2e}, BLER: {bler:.2e}")

    return ber_list, bler_list

# ------------------------------------------------------------
#  Построение графиков
# ------------------------------------------------------------
def plot_curves(snr_db, ber, bler, title="Convolutional Code Performance"):
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db, ber, 'o-', label='BER (soft Viterbi)')
    plt.semilogy(snr_db, bler, 's-', label='BLER')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.title(title)
    plt.legend()
    plt.ylim([1e-6, 1])
    plt.show()

# ------------------------------------------------------------
#  Главная функция
# ------------------------------------------------------------
def main():
    # Параметры симуляции
    snr_points_db = np.arange(0, 7, 1)   # 0, 0.5, 1, ... 6.5 dB
    num_frames = 1000
    frame_len = 500       # информационных бит на кадр

    # Создаём кодер и декодер
    encoder = ConvEncoder(poly1=0o133, poly2=0o171, memory=6)
    decoder = ViterbiDecoder(poly1=0o133, poly2=0o171, memory=6, decoding_type='soft')

    print("Запуск симуляции...")
    ber, bler = simulate_awgn(encoder, decoder, snr_points_db, num_frames, frame_len)

    print("\nРезультаты:")
    for s, b, bl in zip(snr_points_db, ber, bler):
        print(f"SNR {s:4.1f} dB : BER = {b:.2e}   BLER = {bl:.2e}")

    plot_curves(snr_points_db, ber, bler, title="Convolutional Code (133,171) in AWGN, R=1/2")

if __name__ == "__main__":
    main()