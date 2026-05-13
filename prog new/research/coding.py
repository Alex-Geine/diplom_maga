import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

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
            # Здесь нужна реализация _gen_gray, но для BPSK не требуется
            # Заглушка, чтобы не вызывать ошибку при инициализации
            self.bits = None
        # ... остальные модуляции можно добавить

    def demap(self, rx_symbols, noise_variance):
        rx_symbols = np.asarray(rx_symbols).flatten()
        if self.modulation == 'BPSK':
            return 2 * np.real(rx_symbols) / noise_variance
        # Для остальных модуляций – универсальный max-log (если реализован)
        raise NotImplementedError("Для данной модуляции реализуйте demap")

# ---------- Рейт-матчер (повторение / выкалывание) ----------
class RateMatcher:
    """
    Согласование длины кодированного блока с ёмкостью канала.
    Поддерживает повторение (если блок короче) и выкалывание (если длиннее).
    """
    def __init__(self, capacity_bits):
        """
        :param capacity_bits: целевая длина (количество бит, которое можно передать)
        """
        self.capacity_bits = capacity_bits

    def rate_match(self, coded_bits):
        """Приводит длину coded_bits к capacity_bits."""
        n = len(coded_bits)
        target = self.capacity_bits
        if n == target:
            return coded_bits
        elif n < target:
            # Повторение
            reps = target // n
            rem = target % n
            return np.tile(coded_bits, reps + 1)[:target]
        else:
            # Выкалывание (puncturing) – равномерно удаляем лишние биты
            indices = np.round(np.linspace(0, n-1, target)).astype(int)
            return coded_bits[indices]

    def rate_dematch(self, llr_seq):
        """Обратное преобразование LLR после рейт-матчинга."""
        n = len(llr_seq)
        target = self.capacity_bits  # original coded length? Но мы не знаем оригинал.
        # В реальности нужно знать исходную длину до rate_match, но для простоты предположим,
        # что target – это длина, которую мы ожидаем на входе декодера.
        # В данном скрипте мы будем передавать original_len отдельно.
        # Упрощённо: если принятая длина больше – комбинируем повторённые LLR,
        # если меньше – вставляем нули на места выколотых битов.
        # Для наглядности реализуем два случая по признаку "было ли повторение/выкалывание".
        # Для простоты: предполагаем, что мы знаем, какой процесс применялся.
        # В нашем случае мы будем хранить исходную длину coded_bits до рейт-матчинга.
        # Поэтому добавим параметр original_len в метод.
        raise NotImplementedError("Используйте версию с original_len")

    def rate_dematch_with_original_len(self, llr_seq, original_len):
        """
        Восстанавливает LLR последовательность до исходной длины original_len.
        :param llr_seq: LLR после канала (длина = capacity_bits)
        :param original_len: исходная длина кодированного блока до рейт-матчинга
        """
        n_recv = len(llr_seq)
        n_orig = original_len
        if n_recv == n_orig:
            return llr_seq
        elif n_recv > n_orig:
            # Было повторение: объединяем LLR повторённых блоков
            repeats = n_recv // n_orig
            remainder = n_recv % n_orig
            combined = np.zeros(n_orig)
            counts = np.zeros(n_orig)
            for i in range(repeats):
                combined += llr_seq[i*n_orig:(i+1)*n_orig]
                counts += 1
            if remainder > 0:
                combined[:remainder] += llr_seq[repeats*n_orig:]
                counts[:remainder] += 1
            return combined / counts
        else:
            # Было выкалывание: вставляем нулевые LLR на место удалённых битов
            # Находим индексы, которые были сохранены (пропорциональное распределение)
            indices = np.round(np.linspace(0, n_orig-1, n_recv)).astype(int)
            dematched = np.zeros(n_orig)
            for i, idx in enumerate(indices):
                dematched[idx] = llr_seq[i]
            return dematched

# ---------- Блочный перемежитель ----------
class Interleaver:
    def __init__(self, block_size, seed=42):
        self.block_size = block_size
        self.rng = RandomState(seed)
        self.interleaver_pattern = self.rng.permutation(block_size)
        self.deinterleaver_pattern = np.argsort(self.interleaver_pattern)

    def interleave(self, bits):
        bits = np.asarray(bits)
        orig_len = len(bits)
        pad_len = (self.block_size - orig_len % self.block_size) % self.block_size
        if pad_len > 0:
            bits = np.pad(bits, (0, pad_len), constant_values=0)
        bits_reshaped = bits.reshape(-1, self.block_size)
        interleaved = bits_reshaped[:, self.interleaver_pattern].ravel()
        return interleaved, orig_len, pad_len

    def deinterleave(self, bits, orig_len, pad_len):
        bits = np.asarray(bits)
        bits_reshaped = bits.reshape(-1, self.block_size)
        deinterleaved = bits_reshaped[:, self.deinterleaver_pattern].ravel()
        if pad_len > 0:
            deinterleaved = deinterleaved[:orig_len]
        return deinterleaved

# ---------- Свёрточный кодер ----------
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
        for _ in range(self.memory):
            reg = (reg << 1) & 0x7F
            out1 = bin(reg & self.poly1).count('1') & 1
            out2 = bin(reg & self.poly2).count('1') & 1
            out.extend([out1, out2])
        return np.array(out, dtype=int)

# ---------- Декодер Витерби ----------
class ViterbiDecoder:
    def __init__(self, poly1=0o133, poly2=0o171, memory=6):
        self.poly1 = poly1
        self.poly2 = poly2
        self.memory = memory
        self.num_states = 1 << memory
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

# ---------- Симуляция с рейт-матчингом ----------
def simulate_with_interleaver_and_rm(encoder, decoder, interleaver, demapper, rate_matcher,
                                     snr_db, frame_len, capacity_bits,
                                     min_errors=150, max_bits=int(5e6)):
    """
    frame_len – количество информационных бит
    capacity_bits – количество бит, которое можно передать (ресурсная ёмкость)
    """
    snr_lin = 10**(snr_db/10)
    code_rate = 0.5
    es_n0 = snr_lin * code_rate
    noise_var = 1.0 / (2.0 * es_n0)

    total_bits = 0
    total_errors = 0
    trials = 0
    max_trials = 1e4

    while total_errors < min_errors and total_bits < max_bits and trials < max_trials:
        info = np.random.randint(0, 2, frame_len)
        coded = encoder.encode(info)                     # длина = 2*frame_len + 12
        # Рейт-матчинг (согласование с capacity_bits)
        rate_matched = rate_matcher.rate_match(coded)    # длина = capacity_bits
        # Перемежение
        inter, orig_len, pad = interleaver.interleave(rate_matched)
        # BPSK модуляция
        tx = 1 - 2 * inter.astype(float)
        tx = tx.astype(complex)
        # Комплексный AWGN
        noise = np.sqrt(noise_var) * (np.random.randn(len(tx)) + 1j*np.random.randn(len(tx)))
        rx = tx + noise
        # Мягкая демодуляция
        llr = demapper.demap(rx, noise_var)              # длина len(tx) = capacity_bits (после перемежения)
        # Деперемежение
        deinter_llr = interleaver.deinterleave(llr, orig_len, pad)  # длина = capacity_bits (исходная после RM)
        # Рейт-дематчинг (восстанавливаем длину coded)
        dematched_llr = rate_matcher.rate_dematch_with_original_len(deinter_llr, len(coded))
        # Декодирование
        decoded = decoder.decode(dematched_llr, block_len=frame_len)
        errors = np.sum(info != decoded)
        total_errors += errors
        total_bits += frame_len
        trials += 1

    return total_errors / total_bits if total_bits else 0.0

# ---------- Основная функция ----------
def main():
    # Параметры системы
    snr_db = np.arange(0, 7, 1)
    frame_len = 120               # информационных бит (подобрано под capacity)
    capacity_bits = 280           # фиксированная ёмкость (например, число RE * бит на символ)
    block_size = 280              # блок перемежителя равен capacity_bits (или кратен)

    encoder = ConvEncoder()
    decoder = ViterbiDecoder()
    interleaver = Interleaver(block_size, seed=123)
    demapper = SoftDemapper('BPSK')
    rate_matcher = RateMatcher(capacity_bits)

    # Проверка корректности длин
    test_info = np.zeros(frame_len, dtype=int)
    test_coded = encoder.encode(test_info)
    print(f"Информационных бит: {frame_len}")
    print(f"Кодированных бит (до RM): {len(test_coded)}")
    print(f"После rate matching: {capacity_bits}")
    print(f"После перемежения (с дополнением): {interleaver.block_size}")
    print(f"Ожидаемая скорость кода: {frame_len / capacity_bits:.3f} (теоретически 0.5)")

    ber_list = []
    for snr in snr_db:
        ber = simulate_with_interleaver_and_rm(encoder, decoder, interleaver, demapper,
                                               rate_matcher, snr, frame_len, capacity_bits,
                                               min_errors=150, max_bits=int(5e6))
        ber_list.append(ber)
        print(f"{snr:3.1f} dB   BER = {ber:.2e}")

    plt.semilogy(snr_db, ber_list, 'o-')
    plt.grid(True)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Convolutional code + Rate Matching + Interleaver, BPSK, AWGN')
    plt.ylim([1e-6, 1])
    plt.show()

if __name__ == '__main__':
    main()