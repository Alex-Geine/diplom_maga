import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Константы ---
MU = 3.986e14        # Гравитационный параметр Земли (м^3/с^2)
R_EARTH = 6371000    # Радиус Земли (м)
C = 299792458.0      # Скорость света (м/с)
OMEGA_E = 7.2921e-5  # Угловая скорость Земли (рад/с)

def get_orbit(a, e, i_deg, duration, steps=2000):
    """Моделирование орбиты в инерциальной системе"""
    t = np.linspace(0, duration, steps)
    i = np.radians(i_deg)
    
    # Начальное состояние в перигее
    r_p = a * (1 - e)
    v_p = np.sqrt(MU * (2/r_p - 1/a))
    
    # Состояние: [x, y, z, vx, vy, vz]
    state0 = [r_p, 0, 0, 0, v_p * np.cos(i), v_p * np.sin(i)]
    
    def diff_eq(s, t):
        r_vec = s[:3]
        r_mag = np.linalg.norm(r_vec)
        accel = -MU * r_vec / r_mag**3
        return [s[3], s[4], s[5], accel[0], accel[1], accel[2]]
    
    return t, odeint(diff_eq, state0, t)

def calc_normalized_doppler(t_arr, states, user_lat=55.75):
    """Расчет нормированного доплеровского сдвига (df/f0)"""
    lat = np.radians(user_lat)
    # Положение пользователя в ECEF (упрощенно на меридиане 0 в t=0)
    user_pos_fixed = R_EARTH * np.array([np.cos(lat), 0, np.sin(lat)])
    
    doppler_norm = []
    
    for i, t in enumerate(t_arr):
        # Переход в ECEF (вращение спутника относительно Земли)
        theta = OMEGA_E * t
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Позиция и скорость в инерциальной системе
        pos_i = states[i, :3]
        vel_i = states[i, 3:6]
        
        # Поворачиваем позицию спутника в систему Земли
        pos_e = np.array([
            pos_i[0]*cos_t + pos_i[1]*sin_t,
           -pos_i[0]*sin_t + pos_i[1]*cos_t,
            pos_i[2]
        ])
        
        # Скорость в системе Земли (учитываем переносную скорость вращения)
        vel_e = np.array([
            vel_i[0]*cos_t + vel_i[1]*sin_t + OMEGA_E * pos_e[1],
           -vel_i[0]*sin_t + vel_i[1]*cos_t - OMEGA_E * pos_e[0],
            vel_i[2]
        ])
        
        rel_pos = pos_e - user_pos_fixed
        dist = np.linalg.norm(rel_pos)
        
        # Проверка прямой видимости (над горизонтом)
        if np.dot(rel_pos, user_pos_fixed) > 0:
            v_radial = np.dot(vel_e, rel_pos) / dist
            doppler_norm.append(-v_radial / C)
        else:
            doppler_norm.append(np.nan)
            
    return np.array(doppler_norm)

# --- Расчет ---
t = 43200

t_leo, res_leo = get_orbit(R_EARTH + 500000, 0.001, 51.6, t) # 1.5 часа
t_heo, res_heo = get_orbit(26600000, 0.74, 63.4, t)       # 12 часов

doppler_leo = calc_normalized_doppler(t_leo, res_leo) * 1e6   # в ppm
doppler_heo = calc_normalized_doppler(t_heo, res_heo) * 1e6   # в ppm

# --- График ---
plt.figure(figsize=(10, 6))
plt.plot(t_leo/3600, doppler_leo, 'r', label='LEO (500 км) - Один пролет')
plt.plot(t_heo/3600, doppler_heo, 'g', label='HEO (Molniya) - Полцикла')

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.title("Нормированный доплеровский сдвиг ($\Delta f / f_0$)")
plt.xlabel("Время (LEO в мин / HEO в часах)")
plt.ylabel("Сдвиг, ppm ($10^{-6}$)")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
