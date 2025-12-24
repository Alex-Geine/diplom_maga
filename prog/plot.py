import numpy as np
import matplotlib.pyplot as plt
from scipy import special

SNR_dB_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

ber_theoretical = [5.89872026e-02, 4.22114640e-02, 2.81295963e-02, 1.71588057e-02,
 9.37561353e-03, 4.46540036e-03, 1.79121809e-03, 5.79506112e-04,
 1.43180831e-04, 2.52204213e-05, 2.90408116e-06]

ber_0 = [7.91870117e-02, 5.70190430e-02, 3.75976562e-02, 2.37670898e-02,
 1.31713867e-02, 5.55419922e-03, 2.12402344e-03, 6.10351562e-04,
 1.83105469e-04, 4.88281250e-05, 0.00000000e+00]

ber_100 = [7.90771484e-02, 5.65368652e-02, 3.70727539e-02, 2.32238770e-02,
 1.25854492e-02, 5.73730469e-03, 2.52075195e-03, 7.69042969e-04,
 2.13623047e-04, 3.05175781e-05, 0.00000000e+00]

ber_1000 = [7.92785645e-02, 5.62316895e-02, 3.75671387e-02, 2.32238770e-02,
 1.28051758e-02, 6.26831055e-03, 2.56652832e-03, 8.17871094e-04,
 2.71606445e-04, 2.44140625e-05, 6.10351563e-06]

ber_4000 = [8.25012207e-02, 6.00860596e-02, 4.16198730e-02, 2.69409180e-02,
 1.55670166e-02, 8.58459473e-03, 4.13818359e-03, 1.83715820e-03,
 6.86645508e-04, 2.22778320e-04, 7.01904297e-05]

ber_8000 = [0.09310303, 0.07112732, 0.05410461, 0.03825989, 0.02649231, 0.01864929,
 0.01256104, 0.00859985, 0.0057373,  0.00409241, 0.00289917]



"""
    Построение графика сравнения смоделированной и теоретической BER
"""
plt.figure(figsize=(12, 8))
    
# Настройка стилей для смоделированных данных (разные маркеры, серый цвет)
plt.semilogy(SNR_dB_values, ber_0, 
            color='black', linestyle='-', marker='o', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Экспериментальная кривая (V = 0 м/c)')

plt.semilogy(SNR_dB_values, ber_4000, 
            color='black', linestyle='-', marker='s', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Экспериментальная кривая (V = 4000 м/c)')

plt.semilogy(SNR_dB_values, ber_8000, 
            color='black', linestyle='-', marker='^', linewidth=2, 
            markersize=8, markerfacecolor='white', label=f'Экспериментальная кривая (V = 8000 м/c)')

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

filename = f'ber_ofdm_qam_16.png'
filenamePdf = f'ber_ofdm_qam_16.pdf'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.savefig(filenamePdf, dpi=300, bbox_inches='tight')