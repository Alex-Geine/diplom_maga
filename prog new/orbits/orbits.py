import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Константы ---
G = 6.67430e-11  # Гравитационная постоянная
M = 5.972e24     # Масса Земли (кг)
R_EARTH = 6371000 # Средний радиус Земли (м)
MU = G * M       # Гравитационный параметр

def orbit_equations(state, t):
    """Дифференциальные уравнения движения (закон всемирного тяготения)"""
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    # Ускорение a = -mu * r / r^3
    ax, ay, az = -MU * np.array([x, y, z]) / r**3
    return [vx, vy, vz, ax, ay, az]

def get_initial_state(a, e, i_deg):
    """Расчет начального состояния в перигее (упрощенно)"""
    i = np.radians(i_deg)
    r_p = a * (1 - e)               # Расстояние в перигее
    v_p = np.sqrt(MU * (2/r_p - 1/a)) # Скорость в перигее
    
    # Положение на оси X, скорость направлена по Y и Z (в зависимости от наклонения)
    return [r_p, 0, 0, 0, v_p * np.cos(i), v_p * np.sin(i)]

# --- Параметры для настройки ---
# 1. LEO (например, МКС): высота ~400 км, почти круг
a_leo = R_EARTH + 400000 
e_leo = 0.001
i_leo = 51.6

# 2. HEO (типа "Молния"): вытянутая эллиптическая орбита
a_heo = 26600000 
e_heo = 0.74
i_heo = 63.4

# Время моделирования (в секундах) - здесь 24 часа
t = np.linspace(0, 24 * 3600, 5000)

# --- Расчет ---
state_leo = odeint(orbit_equations, get_initial_state(a_leo, e_leo, i_leo), t)
state_heo = odeint(orbit_equations, get_initial_state(a_heo, e_heo, i_heo), t)

# --- Визуализация ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Рисуем Землю (сферу)
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = R_EARTH * np.cos(u) * np.sin(v)
y_e = R_EARTH * np.sin(u) * np.sin(v)
z_e = R_EARTH * np.cos(v)
ax.plot_wireframe(x_e, y_e, z_e, color="royalblue", alpha=0.3, label='Земля')

# 2. Рисуем орбиты
ax.plot(state_leo[:, 0], state_leo[:, 1], state_leo[:, 2], label='LEO (Низкая)', color='red')
ax.plot(state_heo[:, 0], state_heo[:, 1], state_heo[:, 2], label='HEO (Высокая)', color='green')

# Оформление
ax.set_xlabel('X (км)')
ax.set_ylabel('Y (км)')
ax.set_zlabel('Z (км)')
ax.legend()
plt.title('Моделирование спутниковых орбит (LEO vs HEO)')

# Масштабирование осей для наглядности (по самой большой орбите)
limit = a_heo * (1 + e_heo)
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)

plt.show()
