import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

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
    time_indices = np.arange(fft_size)
    doppler_phase = 2 * np.pi * subc_freq * doppler_factor * time_indices
    doppler_signal = np.exp(1j * doppler_phase)

    return doppler_signal

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

# Calculate number of failed bits
def calcFailedBits(inBits, outBits):
    numBits = 0

    for i in range(len(inBits)):
        numBits += 0 if inBits[i] == outBits[i] else 1

    return numBits 


def calculatePoint(fft_size, snr, num_experiments, doppler_exp):
    total_bits = fft_size * num_experiments
    total_errors = 0


    # Num of experiments cycle
    for i in range(num_experiments):
        bits = genBits(fft_size)

        modBits = bpsk(bits)

        spectrum = np.fft.ifft(modBits, fft_size)

        spectrumDoppler = np.multiply(spectrum, doppler_exp)

        noiseSignal = noiseInsertion(spectrumDoppler, snr)

        signalRx = np.fft.fft(noiseSignal, fft_size)
        
        outBits = bpskDemapper(signalRx)

        total_errors += calcFailedBits(bits, outBits)
    
    return total_errors / total_bits

def plotBer(snr_range, ber_results, theoretical_ber=None):
    """
    Построение графика BER
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'bo-', linewidth=2, markersize=6, label='Simulated BER')
    
    if theoretical_ber is not None:
        plt.semilogy(snr_range, theoretical_ber, 'r--', linewidth=2, label='Theoretical BPSK BER')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR Characteristics')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.show()

def theoreticalBpskBer(snr_db):
    """
    Теоретическая BER для BPSK в AWGN канале
    """
    snr_lin = 10 ** (snr_db / 10.0)
    return 0.5 * special.erfc(np.sqrt(snr_lin))

# main function
def main():
    # Parameters
    fft_size       = 12#2048            # size of fft (num of subc in OFDM)
    subc_freq      = 15000.             # subcarrier distanse [Hz]
    light_vel      = 300000000.         # speed of light (3* 10^8 m/sec)
    rx_vel         = 80.              # speed of reseiver (first cosmic velocity)
    doppler_factor = rx_vel / light_vel # Doppler factor for the signal
    num_experiments= 10000
    
    # Calculate Doppler
    #doppler_exp = [1] * fft_size
    doppler_exp = preCalcDoppler(fft_size, subc_freq, doppler_factor)

    num_snr = 10
    snr_vals = [0] * num_snr
    ber_vals = [0] * num_snr
    ber_theor = [0] * num_snr

    for i in range(num_snr):
        print(f"{i+1}/10")
        snr_vals[i] = i
        ber_vals[i] = calculatePoint(fft_size, snr_vals[i], num_experiments, doppler_exp)
        ber_theor[i] = theoreticalBpskBer(snr_vals[i])

    print(f"snr vals: {snr_vals}")
    print(f"ber vals: {ber_vals}")

    plotBer(snr_vals, ber_vals, ber_theor)


main()

