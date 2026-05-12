#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симуляция системы связи 5G NTN с поддержкой помехоустойчивого кодирования
(свёрточный код + перемежитель + декодер Витерби)

Автор: [Ваше имя]
Дата: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
from numpy.random import RandomState
import os
import datetime
import sys

# ========================= УТИЛИТЫ ВИЗУАЛИЗАЦИИ =========================

def plot_channel(data, title="ИЧХ канала"):
    """Визуализация модуля сигнала или частотной характеристики"""
    plt.figure(figsize=(8, 4))
    plt.stem(np.abs(data) if len(data) < 200 else np.abs(data)[:200], use_line_collection=True)
    plt.title(title)
    plt.ylabel("Амплитуда")
    plt.xlabel("Отсчёт")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_constellation(samples, title="Сигнальное созвездие"):
    """
    Визуализирует созвездие по комплексным отсчетам.
    :param samples: массив комплексных чисел (IQ-отсчеты)
    """
    samples = np.array(samples)
    i_coords = samples.real
    q_coords = samples.imag

    plt.figure(figsize=(7, 7))
    plt.scatter(i_coords, q_coords, color='blue', s=10, alpha=0.5, label='Отсчеты')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ========================= 1. ПАРАМЕТРЫ 5G NTN =========================

cp_table = {
    512:  {'normal': 36, 'extended': 40},
    1024: {'normal': 72, 'extended': 80},
    2048: {'normal': 144, 'extended': 160},
    4096: {'normal': 288, 'extended': 320}
}

def get_cp_length(fft_size, cp_type='normal'):
    """Возвращает длину циклического префикса для заданного FFT size"""
    return cp_table[fft_size][cp_type]

# Таблицы Resource Blocks для разных диапазонов
ls_rb_table = {5: {15:25,30:11}, 10:{15:52,30:24}, 15:{15:79,30:38}, 20:{15:106,30:51}}
ka_rb_table = {50:{60:66,120:32}, 100:{60:132,120:66}, 200:{60:264,120:132}, 400:{120:264}}

def get_rb(band, bw_mhz, scs_khz):
    """Возвращает количество RB для заданной конфигурации"""
    if band == 'L_S': 
        return ls_rb_table.get(bw_mhz, {}).get(scs_khz)
    elif band == 'Ka': 
        return ka_rb_table.get(bw_mhz, {}).get(scs_khz)
    return None

def get_fft_size_from_re(num_re):
    """Подбирает минимальный размер FFT >= num_re (степень двойки)"""
    n = 64
    while n <= num_re: 
        n *= 2
    return n


# ========================= 2. МОДУЛЯЦИЯ / ДЕМОДУЛЯЦИЯ =========================

def bits_per_symbol(mod_type):
    """Возвращает количество бит на символ для заданной модуляции"""
    return {'BPSK':1, 'QPSK':2, '16QAM':4, '64QAM':6, '256QAM':8}[mod_type]

def modulate(bits, mod_type):
    """Модуляция битов в комплексные символы"""
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
    """Жёсткая демодуляция символов в биты"""
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
        thresh = np.arange(-14, 15, 2)
        def slice256(x):
            idx = np.digitize(x, thresh, right=False)
            return [(idx>>3)&1, (idx>>2)&1, (idx>>1)&1, idx&1]
        for i, (x,y) in enumerate(zip(I,Q)):
            ibits = slice256(x)
            qbits = slice256(y)
            bits[8*i:8*i+8] = ibits + qbits
        return bits
    
    else:
        raise ValueError(f"Unknown demodulation: {mod_type}")


# ========================= 3. КОДИРОВАНИЕ: КОМПОНЕНТЫ =========================

# ------------------ SoftDemapper для мягких решений (LLR) ------------------
class SoftDemapper:
    """Демаппер для вычисления LLR (Log-Likelihood Ratio)"""
    
    def __init__(self, modulation):
        self.modulation = modulation
        if modulation == 'BPSK':
            self.bits_per_symbol = 1
            self.constellation = np.array([1, -1], dtype=complex)
            self.bits = np.array([[0], [1]])
        elif modulation == 'QPSK':
            self.bits_per_symbol = 2
            self.constellation = np.array([1+1j, 1-1j, -1-1j, -1+1j], dtype=complex) / np.sqrt(2)
            self.bits = np.array([[0,0],[0,1],[1,1],[1,0]])
        elif modulation == '16QAM':
            self.bits_per_symbol = 4
            r = np.array([-3,-1,1,3], dtype=float)
            self.constellation = np.array([x+1j*y for x in r for y in r], dtype=complex) / np.sqrt(10)
            self.bits = self._gen_gray_16qam()
        elif modulation == '64QAM':
            self.bits_per_symbol = 6
            r = np.array([-7,-5,-3,-1,1,3,5,7], dtype=float)
            self.constellation = np.array([x+1j*y for x in r for y in r], dtype=complex) / np.sqrt(42)
            self.bits = self._gen_gray_64qam()
        elif modulation == '256QAM':
            self.bits_per_symbol = 8
            r = np.arange(-15, 16, 2, dtype=float)
            self.constellation = np.array([x+1j*y for x in r for y in r], dtype=complex) / np.sqrt(170)
            self.bits = self._gen_gray_256qam()
        else:
            raise ValueError(f"Unsupported modulation for SoftDemapper: {modulation}")

    def _gen_gray_16qam(self):
        """Gray mapping для 16QAM: [b0,b1] для I, [b2,b3] для Q"""
        bits = []
        gray_i = [[0,0],[0,1],[1,1],[1,0]]  # для уровней -3,-1,1,3
        for gi in gray_i:
            for gq in gray_i:
                bits.append(gi + gq)
        return np.array(bits)

    def _gen_gray_64qam(self):
        """Gray mapping для 64QAM"""
        bits = []
        gray_8 = [[0,0,0],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,1,1],[1,0,1],[1,0,0]]
        for gi in gray_8:
            for gq in gray_8:
                bits.append(gi + gq)
        return np.array(bits)

    def _gen_gray_256qam(self):
        """Gray mapping для 256QAM"""
        bits = []
        gray_16 = []
        for i in range(16):
            gray_16.append([(i>>3)&1, (i>>2)&1, (i>>1)&1, i&1])
        for gi in gray_16:
            for gq in gray_16:
                bits.append(gi + gq)
        return np.array(bits)

    def demap(self, rx_symbols, noise_variance):
        """
        Вычисляет LLR для каждого бита.
        :param rx_symbols: принятые комплексные символы (после эквалайзера)
        :param noise_variance: дисперсия шума на одну компоненту (σ²)
        :return: массив LLR значений
        """
        rx_symbols = np.asarray(rx_symbols).flatten()
        
        # Специальный быстрый случай для BPSK
        if self.modulation == 'BPSK':
            return 2 * np.real(rx_symbols) / noise_variance
        
        # Универсальный max-log LLR для остальных модуляций
        if not np.iscomplexobj(rx_symbols):
            rx_symbols = rx_symbols.astype(complex)
        
        num_symbols = len(rx_symbols)
        num_bits = num_symbols * self.bits_per_symbol
        llr = np.zeros(num_bits)
        
        for i, y in enumerate(rx_symbols):
            # Евклидовы расстояния до всех точек созвездия
            dist = np.abs(y - self.constellation)**2
            
            for bit in range(self.bits_per_symbol):
                idx0 = np.where(self.bits[:, bit] == 0)[0]
                idx1 = np.where(self.bits[:, bit] == 1)[0]
                
                d0 = np.min(dist[idx0]) if idx0.size else np.inf
                d1 = np.min(dist[idx1]) if idx1.size else np.inf
                
                # Max-log LLR: ln(P(b=0|y)/P(b=1|y)) ≈ (d1-d0)/(2σ²)
                llr[i*self.bits_per_symbol + bit] = (d0 - d1) / (2 * noise_variance)
        
        return llr


# ------------------ Блочный перемежитель ------------------
class Interleaver:
    """Блочный перемежитель со случайной перестановкой"""
    
    def __init__(self, block_size, seed=42):
        self.block_size = block_size
        self.rng = RandomState(seed)
        self.interleaver_pattern = self.rng.permutation(block_size)
        self.deinterleaver_pattern = np.argsort(self.interleaver_pattern)

    def interleave(self, bits):
        """Перемежение битового массива"""
        bits = np.asarray(bits)
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


# ------------------ Свёрточный кодер (2,1,7) ------------------
class ConvEncoder:
    """Свёрточный кодер с полиномами 0o133/0o171, память=6"""
    
    def __init__(self, poly1=0o133, poly2=0o171, memory=6):
        self.poly1 = poly1
        self.poly2 = poly2
        self.memory = memory
        self.code_rate = 0.5

    def encode(self, bits):
        """Кодирует входные биты, добавляет tail-биты"""
        bits = np.asarray(bits, dtype=int)
        reg = 0
        out = []
        
        for b in bits:
            reg = ((reg << 1) & 0x7F) | b
            out1 = bin(reg & self.poly1).count('1') & 1
            out2 = bin(reg & self.poly2).count('1') & 1
            out.extend([out1, out2])
        
        # Tail: memory нулей для возврата в нулевое состояние
        for _ in range(self.memory):
            reg = (reg << 1) & 0x7F
            out1 = bin(reg & self.poly1).count('1') & 1
            out2 = bin(reg & self.poly2).count('1') & 1
            out.extend([out1, out2])
        
        return np.array(out, dtype=int)


# ------------------ Декодер Витерби (мягкие решения) ------------------
class ViterbiDecoder:
    """Декодер Витерби для свёрточного кода с мягкими входами (LLR)"""
    
    def __init__(self, poly1=0o133, poly2=0o171, memory=6):
        self.poly1 = poly1
        self.poly2 = poly2
        self.memory = memory
        self.num_states = 1 << memory
        
        # Предвычисление переходов треллиса
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
        """
        Декодирует последовательность LLR.
        :param llr_seq: массив LLR значений (длина = 2 * кодированных бит)
        :param block_len: длина исходного информационного блока
        :return: декодированные биты
        """
        N = len(llr_seq) // 2  # число кодированных пар
        num_states = self.num_states
        
        # Инициализация метрик путей
        path_metrics = np.full(num_states, np.inf)
        path_metrics[0] = 0.0
        traceback = []

        # Прямой проход Витерби
        for step in range(N):
            llr1 = llr_seq[2*step]
            llr2 = llr_seq[2*step+1]
            new_metrics = np.full(num_states, np.inf)
            survivors = [None] * num_states
            
            for s in range(num_states):
                if np.isinf(path_metrics[s]):
                    continue
                for inp in (0, 1):
                    ns, o1, o2 = self.transition[s][inp]
                    # Метрика: -LLR * (1-2*out) для мягкого решения
                    metric = - (llr1*(1-2*o1) + llr2*(1-2*o2))
                    cand = path_metrics[s] + metric
                    if cand < new_metrics[ns]:
                        new_metrics[ns] = cand
                        survivors[ns] = (s, inp)
            
            path_metrics = new_metrics
            traceback.append(survivors)

        # Обратный проход (traceback)
        best_state = np.argmin(path_metrics)
        decoded = []
        cur_state = best_state
        
        for step in range(N-1, -1, -1):
            if traceback[step][cur_state] is None:
                break
            prev_state, inp_bit = traceback[step][cur_state]
            decoded.append(inp_bit)
            cur_state = prev_state
        
        decoded.reverse()
        return np.array(decoded[:block_len], dtype=int)


# ========================= 4. OFDM ПЕРЕДАТЧИК / ПРИЁМНИК =========================

class OFDMTx:
    """OFDM передатчик с опциональным кодированием"""
    
    def __init__(self, num_re, fft_size, cp_type='normal', mod_type='QPSK',
                 use_coding=False, encoder=None, interleaver=None):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2
        self.mod_type = mod_type
        self.bps = bits_per_symbol(mod_type)
        
        # Параметры кодирования
        self.use_coding = use_coding
        self.encoder = encoder
        self.interleaver = interleaver
        self.code_rate = 0.5 if (use_coding and encoder) else 1.0
        
        self.info_bits_per_symbol = self.bps * self.code_rate
        self._orig_len = None
        self._pad_len = None

    def transmit(self, info_bits):
        """
        Преобразует информационные биты в временной сигнал.
        :param info_bits: массив бит до кодирования
        :return: временной сигнал (частотная область, без IFFT для ускорения)
        """
        # Кодирование и перемежение
        if self.use_coding and self.encoder and self.interleaver:
            coded = self.encoder.encode(info_bits)
            inter_bits, self._orig_len, self._pad_len = self.interleaver.interleave(coded)
            bits_to_mod = inter_bits
        else:
            bits_to_mod = info_bits
            self._orig_len = len(info_bits)
            self._pad_len = 0
        
        # Модуляция
        symbols = modulate(bits_to_mod, self.mod_type)
        
        # Размещение на поднесущих (с защитными интервалами по краям)
        freq = np.zeros(self.fft_size, dtype=complex)
        freq[self.offset:self.offset+self.num_re] = symbols
        
        # IFFT (закомментировано для ускорения симуляции - канал линейный в частотной области)
        # time_signal = np.fft.ifft(freq, norm='ortho')
        time_signal = freq
        
        # Добавление циклического префикса (опционально)
        # return np.concatenate([time_signal[-self.cp_len:], time_signal])
        return time_signal


class OFDMRx:
    """OFDM приёмник с опциональным декодированием"""
    
    def __init__(self, num_re, fft_size, cp_type='normal', mod_type='QPSK',
                 use_coding=False, decoder=None, interleaver=None, demapper=None):
        self.num_re = num_re
        self.fft_size = fft_size
        self.cp_len = get_cp_length(fft_size, cp_type)
        self.offset = (fft_size - num_re) // 2
        self.mod_type = mod_type
        self.bps = bits_per_symbol(mod_type)
        
        # Параметры декодирования
        self.use_coding = use_coding
        self.decoder = decoder
        self.interleaver = interleaver
        self.demapper = demapper
        self.code_rate = 0.5 if (use_coding and decoder) else 1.0

    def receive(self, rx_signal, H_freq, noise_variance, return_llr=False):
        """
        Обрабатывает принятый сигнал.
        :param rx_signal: принятый сигнал (частотная область)
        :param H_freq: частотная характеристика канала
        :param noise_variance: дисперсия шума на компоненту (σ²)
        :param return_llr: если True, возвращает LLR вместо жёстких битов
        :return: биты или LLR, и оценка канала
        """
        # Извлечение активных поднесущих
        Y_re = rx_signal[self.offset:self.offset + self.num_re]
        H_re = H_freq[self.offset:self.offset + self.num_re]

        # MMSE эквалайзер
        alfa = 5  # коэффициент запаса для устойчивости
        denominator = np.abs(H_re)**2 + noise_variance * alfa
        W = np.conj(H_re) / denominator
        X_hat = W * Y_re

        if self.use_coding and self.demapper and return_llr:
            # Мягкая демодуляция → LLR
            llr = self.demapper.demap(X_hat, noise_variance=noise_variance)
            
            # Деперемежение LLR
            if self.interleaver and self._orig_len is not None:
                # Вычисляем параметры для деперемежения
                coded_len = len(llr)
                # Для простоты: предполагаем, что блок = вся последовательность
                deinter_llr = self.interleaver.deinterleave(
                    llr, orig_len=coded_len, pad_len=0
                )
                return deinter_llr, H_re
            return llr, H_re
        else:
            # Жёсткая демодуляция (режим без кодирования)
            bits = demodulate(X_hat, self.mod_type)
            return bits, H_re


# ========================= 5. TDL КАНАЛ ДЛЯ 5G NTN =========================

# Профили TDL из 3GPP TR 38.901 (задержки в нс, мощности в дБ)
TDL_PROFILES = {
    'A': (  # TDL-A: NLOS, нелинейный профиль
        [0, 30, 70, 90, 110, 190, 410, 530, 750, 1070, 1090, 1290],
        [-13.4, -0.0, -9.2, -7.5, -4.7, -11.0, -4.7, -16.6, -11.8, -13.4, -19.4, -24.8]
    ),
    'B': (  # TDL-B: средний NLOS
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        [-7.8, -6.2, -7.2, -8.6, -7.5, -10.0, -8.7, -11.0, -11.2, -12.8, -13.4, -14.5, -15.2, -16.9, -17.2, -18.0, -19.4, -20.7, -21.3, -26.4]
    ),
    'C': (  # TDL-C: LOS, сильный первый луч
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        [-3.5, -6.8, -8.2, -9.2, -10.2, -11.1, -12.2, -13.2, -14.1, -15.1, -17.6, -18.7, -19.7, -20.7, -21.7, -22.7, -23.7, -24.7, -25.7, -26.7]
    )
}

class TDLChannel:
    """Канал с задержками (TDL) для моделирования 5G NTN"""
    
    def __init__(self, profile_name, fft_size, scs_khz, d_km, fc_ghz, shadowing_std_db, cp_len):
        """
        :param profile_name: 'A', 'B' или 'C'
        :param fft_size: размер БПФ
        :param scs_khz: шаг поднесущих в кГц
        :param d_km: расстояние (высота орбиты) в км
        :param fc_ghz: несущая частота в ГГц
        :param shadowing_std_db: стандартное отклонение теневых потерь (дБ)
        :param cp_len: длина циклического префикса
        """
        self.fft_size = fft_size
        self.scs_hz = scs_khz * 1e3
        self.fs = fft_size * self.scs_hz
        self.Ts = 1 / self.fs
        self.d_m = d_km * 1e3
        self.fc = fc_ghz * 1e9
        self.c = 3e8
        self.lambda_c = self.c / self.fc
        self.shadowing_std_db = shadowing_std_db
        
        # Потери свободного пространства
        self.Pd_lin = (4 * np.pi * self.d_m / self.lambda_c) ** 2
        self.Pd_db = 10 * np.log10(self.Pd_lin)
        self.cp_len = cp_len

        # Загрузка профиля TDL
        delays_ns, powers_db = TDL_PROFILES[profile_name]
        self.delays_samples = np.round(np.array(delays_ns) * 1e-9 / self.Ts).astype(int)
        self.path_gains_lin = 10 ** (np.array(powers_db) / 10.0)
        self.max_delay = np.max(self.delays_samples)
        self.ir_len = self.max_delay + 1

    def get_path_loss_factor(self):
        """Возвращает коэффициент ослабления с учётом shadowing"""
        Ps_db = np.random.normal(0, self.shadowing_std_db)
        total_loss_db = self.Pd_db + Ps_db
        P_lin = 10 ** (-total_loss_db / 10.0)
        return np.sqrt(P_lin)

    def get_impulse_response_with_delay(self):
        """Генерирует импульсную характеристику канала"""
        path_loss_factor = self.get_path_loss_factor()
        h = np.zeros(self.ir_len, dtype=complex)
        max_power = -np.inf
        time_shift = 0
        
        for gain_lin, delay in zip(self.path_gains_lin, self.delays_samples):
            amp = np.sqrt(gain_lin) * path_loss_factor
            phase = 2 * np.pi * np.random.rand()
            h[delay] += amp * np.exp(1j * phase)
            
            power = gain_lin * (path_loss_factor**2)
            if power > max_power:
                max_power = power
                time_shift = delay
        
        return h, time_shift

    def apply(self, tx_signal, snr_lin):
        """
        Применяет канал к переданному сигналу.
        :param tx_signal: переданный сигнал (частотная область)
        :param snr_lin: отношение сигнал/шум в линейной шкале
        :return: принятый сигнал, H_freq, мощность шума, задержка
        """
        h, time_shift = self.get_impulse_response_with_delay()
        
        # Частотная характеристика канала
        h_full = np.zeros(self.fft_size, dtype=complex)
        h_full[:len(h)] = h
        H_freq = np.fft.fft(h_full, norm='ortho')
        
        # Применение канала (умножение в частотной области)
        X_freq = tx_signal
        Y_freq = X_freq * H_freq
        
        # Добавление шума
        P_signal = np.mean(np.abs(Y_freq)**2)
        N0 = P_signal / snr_lin
        noise = np.sqrt(N0/2) * (np.random.randn(*Y_freq.shape) + 1j*np.random.randn(*Y_freq.shape))
        p_noise = np.mean(np.abs(noise)**2)
        
        rx_signal = Y_freq + noise
        return rx_signal, H_freq, N0, time_shift


# ========================= 6. ТЕОРЕТИЧЕСКИЕ КРИВЫЕ =========================

def ber_awgn_qam(snr_lin, mod_type):
    """Теоретическая BER для AWGN канала"""
    M = 2**bits_per_symbol(mod_type)
    if M == 2:  # BPSK
        return 0.5 * erfc(np.sqrt(snr_lin))
    elif M == 4:  # QPSK
        return 0.5 * erfc(np.sqrt(snr_lin/2))
    else:  # M-QAM
        k = bits_per_symbol(mod_type)
        return (4/k)*(1-1/np.sqrt(M))*0.5*erfc(np.sqrt(3*k*snr_lin/(2*(M-1))))

def bler_theoretical(num_bits, snr_lin, mod_type):
    """Теоретическая BLER (без кодирования)"""
    ber = ber_awgn_qam(snr_lin, mod_type)
    return 1 - (1 - ber)**num_bits

def shannon_capacity(snr_lin):
    """Пропускная способность по Шеннону (бит/с/Гц)"""
    return np.log2(1 + snr_lin)


# ========================= 7. ФУНКЦИИ СИМУЛЯЦИИ =========================

def simulate_snr(snr_db, tx, rx, tdl_channel, num_trials):
    """Симуляция для одного SNR (без кодирования)"""
    snr_lin = 10**(snr_db/10.0)
    total_bit_errors = 0
    block_errors = 0
    bits_per_block = tx.num_re * tx.bps

    for _ in range(num_trials):
        bits_tx = np.random.randint(0, 2, bits_per_block)
        tx_signal = tx.transmit(bits_tx)
        rx_signal, H_freq, N0, time_shift = tdl_channel.apply(tx_signal, snr_lin)
        bits_rx, H_re = rx.receive(rx_signal, H_freq, N0)
        
        # Обрезка до нужной длины
        bits_rx = bits_rx[:bits_per_block]
        errors = np.sum(bits_tx != bits_rx)
        total_bit_errors += errors
        if errors > 0:
            block_errors += 1

    ber = total_bit_errors / (bits_per_block * num_trials)
    bler = block_errors / num_trials
    spectral_eff_max = (tx.bps * tx.num_re) / (tx.fft_size + tx.cp_len)
    throughput = (1 - ber) * spectral_eff_max
    
    return ber, bler, throughput


def simulate_snr_with_coding(snr_db, tx, rx, tdl_channel, num_trials, frame_len=500):
    """Симуляция для одного SNR (с кодированием)"""
    snr_lin = 10**(snr_db/10.0)
    total_bit_errors = 0
    block_errors = 0
    total_info_bits = 0
    
    # Расчёт параметров для демодулятора
    code_rate = tx.code_rate
    es_n0 = snr_lin * code_rate
    noise_var = 1.0 / (2.0 * es_n0)  # дисперсия на компоненту

    for _ in range(num_trials):
        # Генерация информационных битов
        info_bits = np.random.randint(0, 2, frame_len)
        
        # Передача
        tx_signal = tx.transmit(info_bits)
        rx_signal, H_freq, _, time_shift = tdl_channel.apply(tx_signal, snr_lin)
        
        # Приём с мягкими решениями
        llr, H_re = rx.receive(rx_signal, H_freq, noise_var, return_llr=True)
        
        # Декодирование Витерби
        decoded = rx.decoder.decode(llr, block_len=frame_len)
        
        # Подсчёт ошибок
        errors = np.sum(info_bits != decoded)
        total_bit_errors += errors
        total_info_bits += frame_len
        if errors > 0:
            block_errors += 1

    ber = total_bit_errors / total_info_bits if total_info_bits else 1.0
    bler = block_errors / num_trials
    
    # Спектральная эффективность с учётом кода
    spectral_eff_max = (tx.bps * tx.num_re * code_rate) / (tx.fft_size + tx.cp_len)
    throughput = (1 - ber) * spectral_eff_max
    
    return ber, bler, throughput


# ========================= 8. ОСНОВНОЙ СКРИПТ =========================

def main():
    # === Параметры системы ===
    BAND = 'L_S'               # 'L_S' или 'Ka'
    BW_MHZ = 10                # полоса, МГц
    SCS_KHZ = 30               # шаг поднесущих, кГц
    MODULATION = 'BPSK'        # 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'

    # Параметры орбиты и канала
    ORBIT_HEIGHT_KM = 600      # высота в км
    CARRIER_FREQ_GHZ = 2.0     # несущая частота (ГГц)
    SHADOWING_STD_DB = 3.0     # shadowing σ (дБ)
    TDL_PROFILE = 'C'          # 'A', 'B', 'C'

    # Параметры симуляции
    SNR_DB_LIST = np.arange(0, 11, 1)   # диапазон SNR для кодированного режима
    NUM_TRIALS = 200                    # число испытаний на точку
    FRAME_LEN = 500                     # длина информационного блока (бит)
    
    # === Переключатель кодирования ===
    USE_CODING = True  # Установите False для режима без кодирования
    
    # === Расчёт конфигурации ===
    rb = get_rb(BAND, BW_MHZ, SCS_KHZ)
    if rb is None:
        print(f"❌ Ошибка: Неверная комбинация {BAND}/{BW_MHZ}МГц/{SCS_KHZ}кГц")
        sys.exit(1)
    
    num_re = rb * 12
    fft_size = get_fft_size_from_re(num_re)
    cp_type = 'normal'
    
    # === Инициализация компонентов кодирования ===
    if USE_CODING:
        print("🔧 Инициализация компонентов кодирования...")
        encoder = ConvEncoder(poly1=0o133, poly2=0o171, memory=6)
        decoder = ViterbiDecoder(poly1=0o133, poly2=0o171, memory=6)
        
        # Размер блока перемежителя = длина кодированного кадра
        test_coded = encoder.encode(np.zeros(FRAME_LEN))
        interleaver = Interleaver(block_size=len(test_coded), seed=42)
        demapper = SoftDemapper(modulation=MODULATION)
        print(f"   ✓ Код: свёрточный (2,1,7), rate=1/2")
        print(f"   ✓ Блок перемежителя: {len(test_coded)} бит")
    else:
        encoder = decoder = interleaver = demapper = None
        # Для режима без кодирования можно расширить диапазон SNR
        SNR_DB_LIST = np.arange(0, 21, 2)
    
    # === Создание Tx/Rx ===
    tx = OFDMTx(num_re, fft_size, cp_type, MODULATION,
                use_coding=USE_CODING, encoder=encoder, interleaver=interleaver)
    rx = OFDMRx(num_re, fft_size, cp_type, MODULATION,
                use_coding=USE_CODING, decoder=decoder, 
                interleaver=interleaver, demapper=demapper)
    
    # === Создание канала ===
    tdl = TDLChannel(TDL_PROFILE, fft_size, SCS_KHZ,
                     ORBIT_HEIGHT_KM, CARRIER_FREQ_GHZ, SHADOWING_STD_DB, tx.cp_len)

    # === Вывод конфигурации ===
    print("\n" + "="*60)
    print("📡 КОНФИГУРАЦИЯ СИМУЛЯЦИИ 5G NTN")
    print("="*60)
    print(f"{'Кодирование:':<25} {'✅ Включено' if USE_CODING else '❌ Выключено'}")
    print(f"{'Диапазон:':<25} {BAND}")
    print(f"{'Полоса:':<25} {BW_MHZ} МГц")
    print(f"{'SCS:':<25} {SCS_KHZ} кГц")
    print(f"{'Resource Blocks:':<25} {rb}")
    print(f"{'Resource Elements:':<25} {num_re}")
    print(f"{'FFT size:':<25} {fft_size}")
    print(f"{'CP length:':<25} {tx.cp_len} отсчётов")
    print(f"{'Модуляция:':<25} {MODULATION} ({tx.bps} бит/символ)")
    if USE_CODING:
        print(f"{'Инфо. бит/символ:':<25} {tx.info_bits_per_symbol:.2f}")
    print(f"{'Орбита:':<25} {ORBIT_HEIGHT_KM} км")
    print(f"{'Несущая частота:':<25} {CARRIER_FREQ_GHZ} ГГц")
    print(f"{'Shadowing σ:':<25} {SHADOWING_STD_DB} дБ")
    print(f"{'Профиль TDL:':<25} {TDL_PROFILE}")
    print(f"{'SNR диапазон:':<25} {SNR_DB_LIST[0]}..{SNR_DB_LIST[-1]} дБ")
    print(f"{'Испытаний на точку:':<25} {NUM_TRIALS}")
    if USE_CODING:
        print(f"{'Длина блока:':<25} {FRAME_LEN} бит")
    print("="*60 + "\n")

    # === Запуск симуляции ===
    results = {}
    for snr in SNR_DB_LIST:
        print(f"⏳ SNR = {snr:2d} дБ ...", end=" ", flush=True)
        
        if USE_CODING:
            ber, bler, thr = simulate_snr_with_coding(
                snr, tx, rx, tdl, NUM_TRIALS, frame_len=FRAME_LEN)
        else:
            ber, bler, thr = simulate_snr(snr, tx, rx, tdl, NUM_TRIALS)
        
        results[snr] = {'ber': ber, 'bler': bler, 'throughput': thr}
        print(f"BER={ber:.2e}, BLER={bler:.3f}")

    # === Теоретические кривые (для сравнения) ===
    snr_lin_vals = 10**(np.array(SNR_DB_LIST)/10)
    
    if USE_CODING:
        # Для кодированного режима: теоретическая кривая с учётом code rate
        bits_per_block = FRAME_LEN
        bler_theory = bler_theoretical(bits_per_block, snr_lin_vals * code_rate, MODULATION)
    else:
        bits_per_block = num_re * tx.bps
        bler_theory = bler_theoretical(bits_per_block, snr_lin_vals, MODULATION)
    
    shannon = shannon_capacity(snr_lin_vals)
    max_se = (tx.bps * num_re * tx.code_rate) / (fft_size + tx.cp_len)

    # === Построение графиков ===
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # График 1: BLER vs SNR
    plt.figure(figsize=(9, 6))
    bler_sim = [results[s]['bler'] for s in SNR_DB_LIST]
    plt.semilogy(SNR_DB_LIST, bler_sim, 'bo-', linewidth=2, markersize=6, 
                 label='Симуляция (TDL+MMSE)' + ('+Код' if USE_CODING else ''))
    plt.semilogy(SNR_DB_LIST, bler_theory, 'r--', linewidth=1.5, 
                 label='Теория (AWGN)')
    plt.xlabel('SNR, дБ', fontsize=11)
    plt.ylabel('BLER', fontsize=11)
    title = f'BLER: {MODULATION}, TDL-{TDL_PROFILE}, {ORBIT_HEIGHT_KM} км'
    if USE_CODING:
        title += ', Код (2,1,7) rate=1/2'
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    bler_fig = plt.gcf()

    # График 2: Throughput vs SNR
    plt.figure(figsize=(9, 6))
    thr_sim = [results[s]['throughput'] for s in SNR_DB_LIST]
    plt.plot(SNR_DB_LIST, thr_sim, 'bo-', linewidth=2, markersize=6, 
             label=f'{MODULATION} (TDL+MMSE)' + ('+Код' if USE_CODING else ''))
    plt.plot(SNR_DB_LIST, shannon, 'r--', linewidth=1.5, label='Ёмкость Шеннона')
    plt.axhline(y=max_se, color='gray', linestyle=':', linewidth=1.5, 
                label=f'Макс. SE = {max_se:.3f} бит/с/Гц')
    plt.xlabel('SNR, дБ', fontsize=11)
    plt.ylabel('Throughput, бит/с/Гц', fontsize=11)
    plt.title(f'Спектральная эффективность, {BAND} диапазон', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    thr_fig = plt.gcf()

    # === Сохранение результатов ===
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    coding_tag = "CODED" if USE_CODING else "UNCODED"
    save_dir = f"NTN_{MODULATION}_{coding_tag}_{TDL_PROFILE}_H{ORBIT_HEIGHT_KM}km_{date_str}"
    os.makedirs(save_dir, exist_ok=True)
    
    bler_fig.savefig(os.path.join(save_dir, "BLER_vs_SNR.png"), dpi=150, bbox_inches='tight')
    thr_fig.savefig(os.path.join(save_dir, "Throughput_vs_SNR.png"), dpi=150, bbox_inches='tight')
    plt.close('all')

    # Сохранение данных в .mat
    mat_data = {
        'config': {
            'band': BAND, 'bw_mhz': BW_MHZ, 'scs_khz': SCS_KHZ,
            'modulation': MODULATION, 'num_re': num_re, 'fft_size': fft_size,
            'cp_len': tx.cp_len, 'code_rate': tx.code_rate,
            'max_spectral_eff': max_se, 'orbit_km': ORBIT_HEIGHT_KM, 
            'carrier_ghz': CARRIER_FREQ_GHZ, 'shadowing_std_db': SHADOWING_STD_DB, 
            'tdl_profile': TDL_PROFILE, 'use_coding': USE_CODING,
            'frame_len': FRAME_LEN if USE_CODING else None
        },
        'snr_db': np.array(SNR_DB_LIST),
        'bler_sim': np.array([results[s]['bler'] for s in SNR_DB_LIST]),
        'bler_theory_awgn': bler_theory,
        'throughput_sim': np.array([results[s]['throughput'] for s in SNR_DB_LIST]),
        'shannon_capacity': shannon,
        'ber_sim': np.array([results[s]['ber'] for s in SNR_DB_LIST])
    }
    savemat(os.path.join(save_dir, 'simulation_data.mat'), mat_data)

    # === Итоговый вывод ===
    print(f"\n✅ Результаты сохранены в папку: {save_dir}/")
    print("   📄 BLER_vs_SNR.png")
    print("   📄 Throughput_vs_SNR.png") 
    print("   📄 simulation_data.mat")
    
    # Краткая статистика
    print(f"\n📊 Краткие результаты:")
    for snr in [SNR_DB_LIST[0], SNR_DB_LIST[len(SNR_DB_LIST)//2], SNR_DB_LIST[-1]]:
        r = results[snr]
        print(f"   SNR={snr:2d} дБ: BER={r['ber']:.2e}, BLER={r['bler']:.4f}, Thr={r['throughput']:.4f}")

    return results


if __name__ == "__main__":
    results = main()