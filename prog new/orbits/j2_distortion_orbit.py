import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Константы ---
MU = 3.986004418e14
R_E = 6378137.0
J2 = 1.08262668e-3
C = 299792458.0
OMEGA_E = 7.292115e-5

def equations(state, t, use_j2):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Базовая гравитация
    a_g = -MU * np.array([x, y, z]) / r**3
    
    if use_j2:
        z_r2 = (z/r)**2
        factor = 1.5 * J2 * (MU / r**2) * (R_E / r)**2
        a_j2 = factor * np.array([
            (x/r) * (5*z_r2 - 1),
            (y/r) * (5*z_r2 - 1),
            (z/r) * (5*z_r2 - 3)
        ])
        return [vx, vy, vz, a_g[0]+a_j2[0], a_g[1]+a_j2[1], a_g[2]+a_j2[2]]
    
    return [vx, vy, vz, a_g[0], a_g[1], a_g[2]]

def calculate_orbit_doppler(a_km, e, i_deg, use_j2, duration_h, user_lat=55.75):
    a = a_km * 1000
    t = np.linspace(0, duration_h * 3600, 5000)
    i_rad = np.radians(i_deg)
    
    # Начальное состояние (перигей)
    rp = a * (1 - e)
    vp = np.sqrt(MU * (2/rp - 1/a))
    state0 = [rp, 0, 0, 0, vp * np.cos(i_rad), vp * np.sin(i_rad)]
    
    states = odeint(equations, state0, t, args=(use_j2,))
    
    u_lat = np.radians(user_lat)
    user_pos_fixed = R_E * np.array([np.cos(u_lat), 0, np.sin(u_lat)])
    
    doppler = []
    for j, time in enumerate(t):
        th = OMEGA_E * time
        cos_t, sin_t = np.cos(th), np.sin(th)
        
        # Позиция и скорость (Inertial)
        p_i, v_i = states[j, :3], states[j, 3:6]
        
        # В ECEF
        p_e = np.array([p_i[0]*cos_t + p_i[1]*sin_t, -p_i[0]*sin_t + p_i[1]*cos_t, p_i[2]])
        v_e = np.array([v_i[0]*cos_t + v_i[1]*sin_t, -v_i[0]*sin_t + v_i[1]*cos_t, v_i[2]])
        # Относительная скорость с учетом вращения системы координат
        v_rel_vec = v_e - np.cross([0, 0, OMEGA_E], p_e)
        
        rel_pos = p_e - user_pos_fixed
        if np.dot(rel_pos, user_pos_fixed) > 0:
            v_rad = np.dot(v_rel_vec, rel_pos) / np.linalg.norm(rel_pos)
            doppler.append(-v_rad / C * 1e6)
        else:
            doppler.append(np.nan)
            
    return t / 3600, np.array(doppler)

# --- Настройка визуализации ---
orbits = {
    'LEO': (R_E/1000 + 500, 0.001, 51.6, 3),    # 3 часа
    'MEO': (R_E/1000 + 20200, 0.01, 55.0, 12),  # 12 часов
    'HEO': (26600, 0.74, 63.4, 12),             # 12 часов
    'GEO': (42164, 0.0001, 0.05, 24)            # 24 часа
}

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
axes = axes.flatten()

for ax, (name, params) in zip(axes, orbits.items()):
    t_no_j2, d_no_j2 = calculate_orbit_doppler(*params[:3], False, params[3])
    t_j2, d_j2 = calculate_orbit_doppler(*params[:3], True, params[3])
    
    ax.plot(t_no_j2, d_no_j2, 'k--', alpha=0.5, label='Без J2 (Kepler)')
    ax.plot(t_j2, d_j2, label=f'{name} с J2', lw=2)
    
    ax.set_title(f"Доплеровский сдвиг: {name}")
    ax.set_xlabel("Время (ч)")
    ax.set_ylabel("ppm ($10^{-6}$)")
    ax.legend()
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
