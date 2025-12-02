import matplotlib.pyplot as plt
import numpy as np

# Пример: построение sinc-функций для OFDM
t = np.linspace(-4, 4, 1000)
for k in range(-3, 4):
    y = np.sinc(t - k)
    plt.plot(t, y, 'b', linewidth=1.5)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('t')
plt.ylabel('S(t)')
plt.savefig('ofdm_time_df.pdf', bbox_inches='tight')
plt.show()