import random
import numpy as np

# Generate random bits function
def genBits(size):
    bits = [0] * size
    for i in range(size):
        bits[i] = random.randint(0, 1)
    return bits

# BPSK
def bpsk(bits):
    modBits = [0] * len(bits)
    for i in range(len(bits)):
        modBits[i] = complex(1) if bits[i] == 1 else complex(-1)
    return modBits

# Calculate Doppler multyplier per each subc
def preCalcDoppler(fft_size, subc_freq, doppler_factor):
    doppler_exp = [0] * fft_size

    for i in range(fft_size):
        # exp_arg[i] = 2 * Pi * i * subc_freq * doppler_factor
        arg = complex(0, 2 * np.pi * subc_freq * doppler_factor * i)
        doppler_exp[i] = np.exp(arg)

    return doppler_exp

def noiseInsertion(arr, snr_db):
    # Средняя мощность сигнала на отсчёт
    signal_power = np.mean(np.abs(arr) ** 2)
    snr_lin = 10 ** (snr_db / 10.0)
    
    # Мощность шума должна быть signal_power/snr_lin
    noise_power = signal_power / snr_lin

    # Генерируем комплексный шум с единичной дисперсией
    # Для комплексного шума: действительная и мнимая части независимы
    # каждая с дисперсией 1/2, чтобы суммарная дисперсия была 1
    noise = (np.random.randn(len(arr)) + 1j * np.random.randn(len(arr))) / np.sqrt(2)

    # Масштабируем шум до нужной мощности
    noise = noise * np.sqrt(noise_power)
    
    return arr + noise

# Demodulation BPSK
def bpskDemapper(arr):
    outBits = [0] * len(arr)

    for i in range(len(arr)):
        outBits[i] = 1 if arr[i].real > 0 else 0

    return outBits

# Calculate demodulation probality
def calcDemProb(inBits, outBits):
    prob = 0.

    for i in range(len(inBits)):
        prob += 1 if inBits[i] == outBits[i] else 0

    prob /= len(inBits)

    return prob


#def calculatePoint(fft_size, subc_freq, doppler_factor, snr, num_experiments):
#    total_bits = 0
#    total_errors = 0

#    for i in range(num_experiments):
      

# main function
def main():
    # Parameters
    size           = 12#2048               # bits in the signal
    fft_size       = 12#2048               # size of fft (num of subc in OFDM)
    subc_freq      = 15000.             # subcarrier distanse [Hz]
    light_vel      = 300000000.         # speed of light (3* 10^8 m/sec)
    rx_vel         = 8000.              # speed of reseiver (first cosmic velocity)
    doppler_factor = rx_vel / light_vel # Doppler factor for the signal
    snr            = 30                 # Signal to Noise Ratio [dB]

    # Precompute Doppler subcarrier multyplyers
    doppler_exp = preCalcDoppler(fft_size, subc_freq, doppler_factor)

    # 1) Generate bits
    bits = genBits(size)

    # 2) Modulation
    modBits = bpsk(bits)

    # 3) IFFT
    spectrum = np.fft.ifft(modBits, fft_size)

    # 4) We need to insert Doppler shift on each subc
    # i subc: s[i] = s[i] * exp(2 * Pi * j * fd[i]), where fd[i] = i * subc_freq * V / c
    specDoppler = spectrum #np.multiply(spectrum, doppler_exp)

    # 5) AWGN incertion
    noiseSignal = noiseInsertion(specDoppler, snr)

    # 6) FFT
    signalRx = np.fft.fft(noiseSignal, fft_size)

    # 7) Demodulation
    outBits = bpskDemapper(signalRx)

    # 8) Statictics
    persent = calcDemProb(bits, outBits)

    print(f"Output persent: {persent}")

main()

