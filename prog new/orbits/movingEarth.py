import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Константы ---
MU = 3.986e14       # Гравитационный параметр Земли (м^3/с^2)
R_EARTH = 6371000   # Радиус Земли (м)
OMEGA_EARTH = 7.2921159e-5  # Угловая скорость вращения Земли (рад/с)

def orbit_equations(state, t):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    ax, ay, az = -MU * np.array([x, y, z]) / r**3
    return [vx, vy, vz, ax, ay, az]

def to_earth_fixed(state_inertial, t_array):
    """Перевод из инерциальной системы во вращающуюся (ECEF)"""
    state_fixed = np.zeros_like(state_inertial)
    for i, t in enumerate(t_array):
        theta = OMEGA_EARTH * t  # Угол поворота Земли за время t
        # Матрица поворота вокруг оси Z
        c, s = np.cos(theta), np.sin(theta)
        x_i, y_i, z_i = state_inertial[i, 0:3]
        
        state_fixed[i, 0] = x_i * c + y_i * s
        state_fixed[i, 1] = -x_i * s + y_i * c
        state_fixed[i, 2] = z_i
    return state_fixed

# --- Параметры орбит ---
# LEO (МКС-подобная)
a_leo = R_EARTH + 400000
e_leo = 0.001
# HEO (Молния)
a_heo = 26600000
e_heo = 0.74

t = np.linspace(0, 24 * 3600, 10000) # 24 часа

# Решение уравнений (в инерциальной системе)
res_leo_i = odeint(orbit_equations, [a_leo*(1-e_leo), 0, 0, 0, np.sqrt(MU*(2/(a_leo*(1-e_leo)) - 1/a_leo)), 0], t)
res_heo_i = odeint(orbit_equations, [a_heo*(1-e_heo), 0, 0, 0, np.sqrt(MU*(2/(a_heo*(1-e_heo)) - 1/a_heo))*np.cos(np.radians(63.4)), np.sqrt(MU*(2/(a_heo*(1-e_heo)) - 1/a_heo))*np.sin(np.radians(63.4))], t)

# Перевод во вращающуюся систему (относительно поверхности)
res_leo_f = to_earth_fixed(res_leo_i, t)
res_heo_f = to_earth_fixed(res_heo_i, t)

# --- Визуализация ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Отрисовка Земли
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
ax.plot_wireframe(R_EARTH*np.cos(u)*np.sin(v), R_EARTH*np.sin(u)*np.sin(v), R_EARTH*np.cos(v), color="blue", alpha=0.2)

# Отрисовка траекторий относительно поверхности
ax.plot(res_leo_f[:, 0], res_leo_f[:, 1], res_leo_f[:, 2], label='LEO (отн. поверхности)', color='red')
ax.plot(res_heo_f[:, 0], res_heo_f[:, 1], res_heo_f[:, 2], label='HEO (отн. поверхности)', color='green')

ax.set_title("Орбиты во вращающейся системе координат (ECEF)")
ax.legend()
plt.show()
