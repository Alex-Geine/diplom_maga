import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Константы ---
MU      = 3.986e14
R_EARTH = 6371000
C       = 299792458.0
OMEGA_E = 7.2921e-5

def get_orbit_data(a_km, e, i_deg, duration_h):
    """Моделирование доплера относительно неподвижной точки в пространстве"""
    a = a_km * 1000
    t = np.linspace(0, duration_h * 3600, 10000)
    i = np.radians(i_deg)
    
    # Состояние спутника
    rp = a * (1 - e)
    vp = np.sqrt(MU * (2/rp - 1/a))
    state0 = [rp, 0, 0, 0, vp * np.cos(i), vp * np.sin(i)]
    
    def diff_eq(s, t):
        r_vec = s[:3]
        r_mag = np.linalg.norm(r_vec)
        accel = -MU * r_vec / r_mag**3
        return [s[3], s[4], s[5], accel[0], accel[1], accel[2]]
    
    states = odeint(diff_eq, state0, t)
    
    # Фиксированные координаты пользователя в инерциальном пространстве
    user_lat = np.radians(55.75)
    user_pos = R_EARTH * np.array([np.cos(user_lat), 0, np.sin(user_lat)])
    
    doppler_norm = []
    for j in range(len(t)):
        sat_pos = states[j, :3]
        sat_vel = states[j, 3:6]
        
        # Вектор от пользователя к спутнику
        rel_pos = sat_pos - user_pos
        
        # Условие видимости: спутник выше линии горизонта (угол между rel_pos и user_pos < 90°)
        if np.dot(rel_pos, user_pos) > 0:
            # Проекция вектора скорости спутника на вектор дальности
            v_radial = np.dot(sat_vel, rel_pos) / np.linalg.norm(rel_pos)
            doppler_norm.append(-v_radial / C * 1e6)
        else:
            doppler_norm.append(np.nan)
            
    return t / 3600, np.array(doppler_norm)



# --- Параметры ---
orbits = {
    'LEO (500 km)': (R_EARTH/1000 + 500, 0.001, 51.6),
    'MEO (GPS, 20200 km)': (R_EARTH/1000 + 20200, 0.01, 55.0),
    'HEO (Molniya)': (26600, 0.74, 63.4),
    'GEO (35786 km)': (R_EARTH/1000 + 35786, 0.0001, 0.0)
}

plt.figure(figsize=(12, 7))
for name, params in orbits.items():
    t_h, doppler = get_orbit_data(*params, duration_h=12)
    plt.plot(t_h, doppler, label=name, lw=2)

plt.axhline(0, color='black', ls='--', alpha=0.3)
plt.title("Нормированный Доплеровский сдвиг для разных типов орбит")
plt.xlabel("Время (часы)")
plt.ylabel("$\Delta f / f_0$ (ppm, $10^{-6}$)")
plt.legend()
plt.grid(True, alpha=0.2)
plt.ylim(-30, 30)
plt.show()
