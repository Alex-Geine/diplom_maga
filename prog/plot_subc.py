import numpy as np
import matplotlib.pyplot as plt
from scipy import special

SNR_dB_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

ber_theoretical = [5.89872026e-02, 4.22114640e-02, 2.81295963e-02, 1.71588057e-02,
 9.37561353e-03, 4.46540036e-03, 1.79121809e-03, 5.79506112e-04,
 1.43180831e-04, 2.52204213e-05, 2.90408116e-06]

ber_64 = [7.88476562e-02, 5.63564453e-02, 3.71181641e-02, 2.26074219e-02,
 1.19355469e-02, 5.74218750e-03, 2.35644531e-03, 7.05078125e-04,
 1.30859375e-04, 2.73437500e-05, 9.76562500e-07]

ber_512 = [7.95273438e-02, 5.72861328e-02, 3.88398437e-02, 2.37226563e-02,
 1.32207031e-02, 6.22851563e-03, 2.77441406e-03, 9.42382812e-04,
 2.57812500e-04, 5.85937500e-05, 8.78906250e-06]

ber_1024 = [8.27851563e-02, 6.05380859e-02, 4.13505859e-02, 2.66875000e-02,
 1.57841797e-02, 8.44628906e-03, 4.07812500e-03, 1.84960937e-03,
 6.97265625e-04, 2.41210937e-04, 8.30078125e-05]

ber_2048 = [0.09310303, 0.07112732, 0.05410461, 0.03825989, 0.02649231, 0.01864929,
 0.01256104, 0.00859985, 0.0057373,  0.00409241, 0.00289917]


"""
    Построение графика сравнения смоделированной и теоретической BER
"""
plt.figure(figsize=(12, 8))
    
# Настройка стилей для смоделированных данных (разные маркеры, серый цвет)
plt.semilogy(SNR_dB_values, ber_64, 
            color='black', linestyle='-', marker='o', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Экспериментальная кривая (N = 64)')

plt.semilogy(SNR_dB_values, ber_512, 
            color='black', linestyle='-', marker='D', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Экспериментальная кривая (N = 512)')

plt.semilogy(SNR_dB_values, ber_1024, 
            color='black', linestyle='-', marker='s', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Экспериментальная кривая (N = 1024)')

plt.semilogy(SNR_dB_values, ber_2048, 
            color='black', linestyle='-', marker='^', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Экспериментальная кривая (N = 2048)')

# Теоретическая BER (пунктир, темно-серый или черный для контраста)
plt.semilogy(SNR_dB_values, ber_theoretical, 
            color='black', linestyle='--', linewidth=3, 
            label='Теоретическая кривая')
    
# Настройка графика
plt.xlabel('SNR, дБ', fontsize=24)
plt.ylabel('BER', fontsize=24)
#plt.title(f'Сравнение BER для OFDM системы\nN=2048', fontsize=16)
plt.grid(True, which='both', alpha=0.3, linestyle='--')
plt.legend(fontsize=18)
plt.ylim([1e-4, 1])
plt.tick_params(axis='both', which='major', labelsize=20)  # Основные деления
plt.tick_params(axis='both', which='minor', labelsize=16)  # Промежуточные (мелкие) деления
# Добавление сетки
ax = plt.gca()
ax.grid(True, which='minor', alpha=0.2, linestyle=':')

plt.tight_layout()

filename = f'ber_ofdm_qam_16_subc.png'
filenamePdf = f'ber_ofdm_qam_16_subc.pdf'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.savefig(filenamePdf, dpi=300, bbox_inches='tight')