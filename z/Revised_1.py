import numpy as np
import matplotlib.pyplot as plt


'''

# How to calculate the mean T_j_I and change in delta T_j_I
# Also pf value will be different each 
# Also the new sec will start from old T_j_I and T_j_D and not from start 

'''

f = 50   # grid frequency
Vs = 120 # Grid Voltage
P = 2500 # Active power
M = 1    # Modulation_index
V_dc = 200 # DC Voltage

pf = 1 # power factor
Time_period_sec = 1  # for example


step = 0.001
t = np.arange(0, Time_period_sec + step, step)  # ensures it includes endpoint



if pf == 0:
    S = P
    Is = S / Vs
    P = 0
else:
    S = P / abs(pf)
    Is = S / Vs

if pf < 0:    # Inductive
    phi = np.arccos(abs(pf)) * -1
    Q = np.sqrt(S ** 2 - P ** 2) * -1
elif pf > 0:  # Capacitive
    phi = np.arccos(abs(pf))
    Q = np.sqrt(S ** 2 - P ** 2)
else:
    phi = 0
    Q = S

omega = 2 * np.pi * f


vs_inverter = np.sqrt(2) * Vs * np.sin(omega * t + phi) # Inverter Voltage
is_inverter = np.sqrt(2) * Is * np.sin(omega * t)       # Inverter Current

m = (M * np.sin(omega * t + phi) + 1)/2

is_I = np.sqrt(2) * Is * np.sin(omega * t) * m   # IGBT Current
is_I  = np.maximum(is_I, 0)
is_D = - np.sqrt(2) * Is * np.sin(omega * t) * m # Diode Current
is_D  = np.maximum(is_D, 0)

'''
Impedance
'''

#  Paste

r_paste = np.array([0.10])       # K/W
tau_paste = np.array([1e-4]) # s (very small, so it settles almost instantly)
Z_Paste = np.sum(r_paste * (1 - np.exp(-t[:, None] / tau_paste)), axis=1)



# Impedance Heat Sink

r_sink = np.array([1.3,2.0])
tau_sink= np.array([0.8, 40.0])   # s
Z_Sink = np.sum(r_sink * (1 - np.exp(-t[:, None] / tau_sink)), axis=1)

# Impedance IGBT

r_I = np.array([7.0e-3, 3.736e-2, 9.205e-2, 1.2996e-1, 1.8355e-1])  # [K/W]
tau_I = np.array([4.4e-5, 1.0e-4, 7.2e-4, 8.3e-3, 7.425e-2])        # [s]
Z_IGBT = np.sum(r_I * (1 - np.exp(-t[:, None] / tau_I)), axis=1)

# Impedance Diode

r_D = np.array([4.915956e-2, 2.254532e-1, 3.125229e-1, 2.677344e-1, 1.951733e-1])  # [K/W]
tau_D = np.array([7.5e-6, 2.2e-4, 2.3e-3, 1.546046e-2, 1.078904e-1])              # [s]
Z_DIODE = np.sum(r_D * (1 - np.exp(-t[:, None] / tau_D)), axis=1)


'''
Switching losses
'''


# IGBT

f_sw = 10 * 1000
t_on = 60e-9
t_off = 259e-9

E_on_I = (np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_on
E_off_I = (np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_off
P_sw_I = (E_on_I + E_off_I) * f_sw
P_sw_I  = np.maximum(P_sw_I, 0)

# Diode

I_ref = 30.0   # A, from datasheet test condition
V_ref = 400.0  # V, from datasheet test condition
Err_D = 0.35e-3 # J, diode reverse recovery energy (adjust for temp)

P_sw_D  = ((np.sqrt(2) / np.pi) * (is_D * V_dc) / (I_ref * V_ref))  * Err_D * f_sw
P_sw_D  = np.maximum(P_sw_D, 0)


'''
Conduction losses
'''


# IGBT

R_IGBT = 0.0175
V_0_IGBT = 1.8

P_con_I = (((is_I ** 2 / 4.0) * R_IGBT) + ((is_I / np.sqrt(2 * np.pi)) * V_0_IGBT) +
           ((((is_I ** 2 / 4.0) * (8 * M / (3 * np.pi)) * R_IGBT) + ((is_I / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_IGBT)) * abs(pf)))
P_con_I  = np.maximum(P_con_I, 0)

# Diode

V_0_D = 1.3   # V
R_D  = 0.023 # Ohm

P_con_D = ((((is_D ** 2 / 4.0) * R_D) + ((is_D / np.sqrt(2 * np.pi)) * V_0_D)) +
             ((((is_D ** 2 / 4.0)) * ((8 * M / (3 * np.pi)) * R_D)) + ((is_D / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_D)) * abs(pf))
P_con_D  = np.maximum(P_con_D, 0)


'''
Junction temperatures
'''


Tamb = 25.0
dt = t[1] - t[0]

P_I   = np.maximum(P_sw_I + P_con_I, 0.0)   # IGBT loss
P_D   = np.maximum(P_sw_D + P_con_D, 0.0)   # Diode loss
P_leg = P_I + P_D                               # drives SHARED pad + sink

# RC sets
r_jc_I, tau_jc_I = r_I, tau_I
r_jc_D, tau_jc_D = r_D, tau_D
r_p, tau_p = r_paste, tau_paste
r_s, tau_s = r_sink, tau_sink

# Precompute alphas
alpha_I = np.exp(-dt / tau_jc_I)
alpha_D = np.exp(-dt / tau_jc_D)
alpha_p = np.exp(-dt / tau_p)
alpha_s = np.exp(-dt / tau_s)

# Branch state temps
Tbr_I = np.zeros_like(r_jc_I, dtype=float)
Tbr_D = np.zeros_like(r_jc_D, dtype=float)
Tbr_p = np.zeros_like(r_p,     dtype=float)
Tbr_s = np.zeros_like(r_s,     dtype=float)

# Outputs
T_j_I = np.zeros_like(t, dtype=float)
T_j_D = np.zeros_like(t, dtype=float)
T_shared = np.zeros_like(t, dtype=float)

# Time stepping
for k in range(len(t)):

    # Separate j→c
    Tbr_I = alpha_I*Tbr_I + (1.0 - alpha_I) * (r_jc_I * P_I[k])
    Tbr_D = alpha_D*Tbr_D + (1.0 - alpha_D) * (r_jc_D * P_D[k])

    # Shared pad+sink
    Tbr_p = alpha_p*Tbr_p + (1.0 - alpha_p) * (r_p * P_leg[k])
    Tbr_s = alpha_s*Tbr_s + (1.0 - alpha_s) * (r_s * P_leg[k])

    shared_rise = Tbr_p.sum() + Tbr_s.sum()
    T_shared[k] = shared_rise

    # Junction temps
    T_j_I[k] = Tamb + shared_rise + Tbr_I.sum()
    T_j_D[k] = Tamb + shared_rise + Tbr_D.sum()


plt.figure(figsize=(6.4, 4.8))
plt.plot(t, T_j_I, label="T_j_I")
plt.plot(t, T_j_D, label="T_j_D")
plt.xlabel("Time (s)")
#plt.xlim(max(t)-0.02, max(t))
plt.ylabel("Temp (C)")
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig(f"Figures/pf_{pf}_{Time_period_sec}.png")


# ---- ONE grid period stats (last period in the simulated window) ----
Tgrid = 1.0 / f                           # [s] grid period (20 ms @ 50 Hz)
Ngrid = max(1, int(round(Tgrid / dt)))    # samples per grid period


# Slice of the last electrical period
sl = slice(len(t) - Ngrid, len(t))

# IGBT stats
TjI_mean  = float(np.mean(T_j_I[sl]))                 # \bar{T}_{j,I}
TjI_delta = float(np.max(T_j_I[sl]) - np.min(T_j_I[sl]))  # \Delta T_{j,I}

# Diode stats
TjD_mean  = float(np.mean(T_j_D[sl]))                 # \bar{T}_{j,D}
TjD_delta = float(np.max(T_j_D[sl]) - np.min(T_j_D[sl]))  # \Delta T_{j,D}

print("Mean Tj_I (°C):", TjI_mean, "  ΔTj_I (°C):", TjI_delta)
print("Mean Tj_D (°C):", TjD_mean, "  ΔTj_D (°C):", TjD_delta)