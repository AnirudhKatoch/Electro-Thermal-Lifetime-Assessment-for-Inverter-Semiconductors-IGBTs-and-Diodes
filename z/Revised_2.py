import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import time
import warnings

start = time.perf_counter()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Input Parameters
#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

pf   = np.array([1, 1, 1, 1, 1])      # [-] Inverter power factor with second resolution
P    = np.full(len(pf), 17000, dtype=float)      # [W] Inverter RMS Active power
Q    = np.full(len(pf), 0, dtype=float)         # [W] Inverter RMS Reactive power
Vs   = np.full(len(pf), 120)       # [V] Inverter RMS AC side voltage
V_dc = np.full(len(pf), 200)       # [V] Inverter DC side voltage
f   = 50                                   # [Hz] Grid frequency
M   = 1                                    # [-] Inverter modulation index
Tamb = 298.15                              # [K] Ambient Temperature
Impedance_calculations = False

if (pf[0] == 0 and Q[0] == 0):
    Q[0] = 1e-6
    warnings.warn("Q[0] was zero while pf=0. Setting Q[0] = 1e-6 to avoid invalid state.", UserWarning)
elif (pf[0] != 0 and P[0] == 0):
    P[0] = 1e-6
    warnings.warn("P[0] was zero while pf≠0. Setting P[0] = 1e-6 to avoid invalid state.", UserWarning)

#----------------------------------------#
# Switching losses
#----------------------------------------#

# IGBT

f_sw = 10 * 1000 # [Hz] Inverter switching frequency
t_on = 60e-9     # [s] Effective turn-on time = td(on) + tr ≈ 23 ns + 37 ns (td is delay period and tr is rising time)  [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 5 datasheet]
t_off = 259e-9   # [s] Effective turn-off time = td(off) + tf ≈ 235 ns + 24 ns (td is delay period and tf is fall time) [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 5 datasheet]

# Diode

I_ref = 30.0    # [A] Reference test current for diode reverse recovery  [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 6 datasheet]
V_ref = 400.0   # [V] Reference test voltage for diode reverse recovery  [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 6 datasheet]
Err_D = 0.352e-3 # [J] Reverse recovery energy per switching event       [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 5 datasheet]

#----------------------------------------#
# Conduction losses
#----------------------------------------#

# IGBT

R_IGBT = 0.01466  # [Ohm] Effective on-resistance for conduction model  [Note: Value is temperature and current dependent, author assumes constant current of 50 A and temp of 25°C] [Fig 6 and Page 5 datasheet]
V_0_IGBT = 1.117   # [V]   Effective knee voltage                       [Note: Value is temperature and current dependent, author assumes constant current of 50 A and temp of 25°C] [Fig 6 and Page 5 datasheet]

# Diode

V_0_D = 1.23   # [V]    Effective forward knee voltage [Note: Value is temperature and current dependent, author assumes constant current of 30 A and temp of 25°C] [Fig 28 and Page 5 datasheet]
R_D  = 0.0164  # [Ohm]  Effective dynamic resistance   [Note: Value is temperature and current dependent, author assumes constant current of 30 A and temp of 25°C] [Fig 28 and Page 5 datasheet]

#----------------------------------------#
# Thermal Parameters
#----------------------------------------#

# Paste

# Case-to-sink thermal interface (SIL PAD® TSP-1600)
# Source: Henkel/Bergquist SIL PAD® TSP-1600 datasheet (TIM)

r_paste = np.array([0.10])       # [K/W] Thermal resistance
tau_paste = np.array([1e-4])     # [s]   Thermal time constant

# Heat Sink

# Sink-to-ambient (Aavid/Boyd 6399B, RθSA ≈ 3.3 K/W natural convection)
# Source: Aavid/Boyd 6399B heatsink datasheet (TO-247 natural convection)

r_sink = np.array([1.3,2.0])     # [K/W] Thermal resistance
tau_sink= np.array([0.8, 40.0])  # [s]   Thermal time constant

# IGBT

# Source: Infineon IKW50N60H3 datasheet, Fig. 21 (Foster RC coefficients)

r_I = np.array([7.0e-3, 3.736e-2, 9.205e-2, 1.2996e-1, 1.8355e-1])  # [K/W] Thermal resistance
tau_I = np.array([4.4e-5, 1.0e-4, 7.2e-4, 8.3e-3, 7.425e-2])        # [s]   Thermal time constant

# Diode

# Source: Infineon IKW50N60H3 datasheet, Fig. 22 (Foster RC coefficients)

r_D = np.array([4.915956e-2, 2.254532e-1, 3.125229e-1, 2.677344e-1, 1.951733e-1])  # [K/W] Thermal resistance
tau_D = np.array([7.5e-6, 2.2e-4, 2.3e-3, 1.546046e-2, 1.078904e-1])               # [s]   Thermal time constant

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Calculations
#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

Time_period = len(pf)              # [-]  Number of seconds in simulation (length of pf array)
omega = 2*np.pi*f                  # [rad/s] Angular frequency of the grid (ω = 2πf)
dt    = 0.001                      # [s] Simulation timestep (1 ms)
Tgrid = 1.0 / f                    # [s] Grid period (time for one full cycle at frequency f)
Ngrid = int(round(Tgrid / dt))     # [-] Number of simulation steps per grid cycle (at f = 50 Hz and dt = 1 ms → 20 steps per cycle)
Nsec  = int(round(1.0   / dt))     # [-] Number of simulation steps per second     (with dt = 1 ms → 1000 steps per second)

#----------------------------------------#
# Inverter Power Flow (P, Q, S, I, φ)
#----------------------------------------#

S = np.zeros_like(pf, dtype=float)   # [VA] Inverter RMS apparent power
Is = np.zeros_like(pf, dtype=float)  # [A] Inverter RMS current
phi = np.zeros_like(pf, dtype=float) # [rad] Phase angle

for i in range(len(pf)):
    if pf[i] == 0:
        P[i] = 0
        S[i] = np.sqrt(P[i]**2 + Q[i]**2)
        Is[i] = S[i] / Vs[i]
        if S[i] == 0:
            phi[i] = 0
        else:
            phi[i] = np.pi/2 if Q[i] > 0 else -np.pi/2
    else:
        S[i] = P[i] / abs(pf[i])
        Is[i] = S[i] / Vs[i]
        if pf[i] < 0:  # inductive
            phi[i] = -np.arccos(abs(pf[i]))
            Q[i] = - np.sqrt(S[i] ** 2 - P[i] ** 2)
        else:          # capacitive
            phi[i] = np.arccos(abs(pf[i]))
            Q[i] = np.sqrt(S[i] ** 2 - P[i] ** 2)


#----------------------------------------#
# Impedance
#----------------------------------------#

if Impedance_calculations == True:

    time_Z = np.arange(0.0,len(pf), dt)
    Z_Paste = np.sum(r_paste * (1 - np.exp(-time_Z[:, None] / tau_paste)), axis=1) # [K/W] Case to sink interface (TIM pad)
    Z_Sink = np.sum(r_sink * (1 - np.exp(-time_Z[:, None] / tau_sink)), axis=1)    # [K/W] Sink to ambient path (extrusion heatsink)
    Z_IGBT = np.sum(r_I * (1 - np.exp(-time_Z[:, None] / tau_I)), axis=1)          # [K/W] Junction to case thermal impedance of IGBT chip
    Z_DIODE = np.sum(r_D * (1 - np.exp(-time_Z[:, None] / tau_D)), axis=1)         # [K/W] Junctiont case thermal impedance of diode chip

    fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))
    ax1.plot(time_Z, Z_Paste, label="Paste")
    ax1.plot(time_Z, Z_Sink, label="Sink")
    ax1.plot(time_Z, Z_IGBT, label="IGBT junction to case")
    ax1.plot(time_Z, Z_DIODE, label="Diode junction to case")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Thermal impedance [K/W]")
    ax1.set_xscale("log")
    ax1.grid(True, which="both", linestyle="--")
    ax1.legend()
    #fig1.tight_layout()
    fig1.savefig("Figures/Impedance_values.png")

#----------------------------------------#
# Thermal state & constants
#----------------------------------------#

# Precompute alphas (discrete-time decay factors for RC branches)

alpha_I = np.exp(-dt / tau_I)       # [-] IGBT junction-to-case RC branches (Foster ladder)
alpha_D = np.exp(-dt / tau_D)       # [-] Diode junction-to-case RC branches (Foster ladder)
alpha_p = np.exp(-dt / tau_paste)   # [-] Case-to-sink thermal interface (paste layer / TIM)
alpha_s = np.exp(-dt / tau_sink)    # [-] Sink-to-ambient RC branches (heatsink thermal network)

# Branch state temps

Tbr_I = np.zeros_like(r_I, dtype=float)     # [K] Temperature rise contributions of each Foster RC branch for IGBT junction → case.
Tbr_D = np.zeros_like(r_D, dtype=float)     # [K] Temperature rise contributions of each Foster RC branch for diode junction → case.
Tbr_p = np.zeros_like(r_paste, dtype=float) # [K] Temperature rise across the thermal paste (case → heatsink interface).
Tbr_s = np.zeros_like(r_sink, dtype=float)  # [K] Temperature rise contributions of each Foster RC branch for the heatsink (sink → ambient).

def Inverter_voltage_and_current(Vs,Is,phi,t,omega):

    """

    Compute instantaneous inverter voltage and current waveforms.

    Parameters
    ----------
    Vs : float
        RMS value of inverter AC-side phase voltage [V].
    Is : float
        RMS value of inverter AC-side output current [A].
    phi : float
        Phase angle between voltage and current [rad].
        (negative = current lags voltage = inductive load,
         positive = current leads voltage = capacitive load).
    t : array
        Time instant [s].
    omega : float
        Angular frequency of the grid (ω = 2πf) [rad/s].

    Returns
    -------
    vs_inverter : array
        Instantaneous inverter output voltage [V].
    is_inverter : array
        Instantaneous inverter output current [A].

    """

    vs_inverter = np.sqrt(2) * Vs * np.sin(omega * t + phi)
    is_inverter = np.sqrt(2) * Is * np.sin(omega * t)

    return vs_inverter, is_inverter


def Instantaneous_modulation(M,omega,t,phi):

    """
    Calculate the inverter modulation function m(t).

    Parameters
    ----------
    M : float
        Modulation index of the inverter [-] (typically 0.8–1.0 for PV inverters).
    omega : float
        Angular frequency of the AC grid [rad/s] (ω = 2πf).
    t : array
        Time instant [s].
    phi : float
        Phase angle of the inverter output current relative to the voltage [rad].

    Returns
    -------
    m : float
            Instantaneous modulation function [-].

    """

    m = (M * np.sin(omega * t + phi) + 1) / 2

    return m

def IGBT_and_diode_current(Is, t, m):

    """
    Calculate the instantaneous IGBT and diode currents in one inverter leg

    Parameters
    ----------
    Is : float
        RMS value of the inverter output current [A].
    t : array
        Time instant [s].
    m : array
        Instantaneous modulation function [-], typically between 0 and 1.

    Returns
    -------
    is_I : array
        Instantaneous IGBT current [A]. (Non-negative only, conduction blocked when negative.)
    is_D : array
        Instantaneous diode current [A]. (Non-negative only, conduction blocked when negative.)
    """

    base = np.sqrt(2) * Is * np.sin(omega * t) * m
    is_I = np.maximum(base, 0)
    is_D = np.maximum(-base, 0)

    return is_I, is_D

def Switching_losses(V_dc, is_I, t_on, t_off, f_sw, is_D, I_ref, V_ref, Err_D):

    """
    Calculate the IGBT and diode switching power losses.

    Parameters
    ----------
    V_dc : float
        DC-link voltage of the inverter [V].
    is_I : array
        Instantaneous current through the IGBT [A].
    t_on : float
        Effective IGBT turn-on time [s].
    t_off : float
        Effective IGBT turn-off time [s].
    f_sw : float
        Inverter switching frequency [Hz].
    is_D : array
        Instantaneous current through the diode [A].
    I_ref : float
        Reference test current for diode reverse recovery [A].
    V_ref : float
        Reference test voltage for diode reverse recovery [V].
    Err_D : float
        Reverse recovery energy per switching event for the diode [J].

    Returns
    -------
    P_sw_I : array
        Instantaneous IGBT switching power loss [W].
    P_sw_D : array
        Instantaneous diode switching power loss [W].

    """

    # IGBT

    E_on_I = (np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_on
    E_off_I = (np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_off
    P_sw_I = (E_on_I + E_off_I) * f_sw
    P_sw_I = np.maximum(P_sw_I, 0)

    # Diode

    P_sw_D = ((np.sqrt(2) / np.pi) * (is_D * V_dc) / (I_ref * V_ref)) * Err_D * f_sw
    P_sw_D = np.maximum(P_sw_D, 0)

    return P_sw_I, P_sw_D


def Conduction_losses(is_I,R_IGBT,V_0_IGBT,M,pf,is_D,R_D,V_0_D):

    """
    Calculate conduction losses of the inverter’s IGBT and diode.

    Parameters
    ----------
    is_I : array
        Instantaneous value of the inverter output current flowing through the IGBT [A].
    R_IGBT : float
        Effective on-resistance of the IGBT conduction model [Ohm].
    V_0_IGBT : float
        Effective knee voltage of the IGBT [V].
    M : float
        Modulation index of the inverter [-].
    pf : float
        Power factor of inverter output current [-].
        (negative = inductive load, current lags voltage;
         positive = capacitive load, current leads voltage).
    is_D : array
        Instantaneous value of the inverter output current flowing through the diode [A].
    R_D : float
        Effective dynamic resistance of the diode [Ohm].
    V_0_D : float
        Effective forward knee voltage of the diode [V].

    Returns
    -------
    P_con_I : array
        Instantaneous conduction loss of the IGBT [W].
    P_con_D : array
        Instantaneous conduction loss of the diode [W].
    """

    # IGBT

    P_con_I = (((is_I ** 2 / 4.0) * R_IGBT) + ((is_I / np.sqrt(2 * np.pi)) * V_0_IGBT) +
               ((((is_I ** 2 / 4.0) * (8 * M / (3 * np.pi)) * R_IGBT) + (
                           (is_I / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_IGBT)) * abs(pf)))
    P_con_I = np.maximum(P_con_I, 0)

    # Diode

    P_con_D = ((((is_D ** 2 / 4.0) * R_D) + ((is_D / np.sqrt(2 * np.pi)) * V_0_D)) -
               ((((is_D ** 2 / 4.0)) * ((8 * M / (3 * np.pi)) * R_D)) + ((is_D / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_D)) * abs(pf))
    P_con_D = np.maximum(P_con_D, 0)

    return P_con_I, P_con_D

def ar_per_cycle(Tcyc, dt):

    """
    Calculate the average temperature rise rate (a_r) per cycle

    Parameters
    ----------
    Tcyc : ndarray, shape (ncycles, N)
        Array of junction temperatures for each cycle.
        Each row corresponds to one complete grid cycle
        with N samples [K].
    dt : float
        Simulation timestep [s]

    Returns
    -------
    ar : ndarray, shape
        Average temperature rise rate [K/s] for each cycle.
    t_cycle_heat : ndarray, shape (ncycles,)
        Heating duration [s] from Tmin to Tmax within each cycle.
    """

    ncycles, N = Tcyc.shape
    ar = np.zeros(ncycles)
    t_cycle_heat = np.zeros(ncycles)

    for k in range(ncycles):
        T = Tcyc[k]
        i_min = int(np.argmin(T))
        i_max = int(np.argmax(T))

        # forward sample distance min→max (wrap if needed)
        ds = (i_max - i_min) % N
        if ds == 0:
            ar[k] = 0.0
            t_cycle_heat[k] = 0.0
            continue

        t_rise = ds * dt          # heating duration [s]
        dT = T[i_max] - T[i_min]  # temperature swing [K]

        t_cycle_heat[k] = t_rise
        ar[k] = dT / t_rise

    return ar, t_cycle_heat


def mother_function(dt,       # Input = Float
                    Vs,       # Input = Float, Originally = Array, Will go in loop
                    Is,       # Input = Float, Originally = Array, Will go in loop
                    phi,      # Input = Float, Originally = Array, Will go in loop
                    omega,    # Input = Float
                    M,        # Input = Float
                    V_dc,     # Input = Float, Originally = Array, Will go in loop
                    t_on,     # Input = Float
                    t_off,    # Input = Float
                    f_sw,     # Input = Float
                    I_ref,    # Input = Float
                    V_ref,    # Input = Float
                    Err_D,    # Input = Float
                    R_IGBT,   # Input = Float
                    V_0_IGBT, # Input = Float
                    pf,       # Input = Float, Originally = Array, Will go in loop
                    R_D,      # Input = Float
                    V_0_D,    # Input = Float
                    alpha_I,  # Input = Array
                    alpha_D,  # Input = Array
                    alpha_p,  # Input = Array
                    r_I,      # Input = Array
                    r_D,      # Input = Array
                    r_paste,  # Input = Array
                    r_sink,   # Input = Array
                    Tamb):    # Input = Float

    """
    Simulate one-second electro-thermal behavior of a single inverter leg and return
    electrical waveforms, loss components, and device junction temperatures.

    The function advances a shared thermal state consisting of per-branch Foster
    RC temperatures for the IGBT, diode, paste (case→sink), and heatsink (sink→ambient).
    These states are kept in module-level globals and therefore *persist* across calls from cycle to next.

    Parameters
    ----------
    dt : float
        Simulation time step [s].
    Vs : float
        RMS AC-side phase voltage of the inverter [V].
    Is : float
        RMS AC-side output current of the inverter [A].
    phi : float
        Current–voltage phase angle at the AC side [rad].
        (negative = inductive, current lags; positive = capacitive, current leads)
    omega : float
        Angular grid frequency ω = 2πf [rad/s].
    M : float
        Modulation index (typically 0.8–1.0) [-].
    V_dc : float
        DC-link voltage [V].
    t_on : float
        Effective IGBT turn-on time (td(on)+tr) [s].
    t_off : float
        Effective IGBT turn-off time (td(off)+tf) [s].
    f_sw : float
        Switching frequency [Hz].
    I_ref : float
        Diode reverse-recovery reference current from datasheet [A].
    V_ref : float
        Diode reverse-recovery reference voltage from datasheet [V].
    Err_D : float
        Diode reverse-recovery energy per event at (I_ref, V_ref) [J].
    R_IGBT : float
        Effective IGBT on-resistance for conduction model [Ω].
    V_0_IGBT : float
        Effective IGBT knee voltage for conduction model [V].
    pf : float
        Power factor magnitude used in conduction-loss expressions [-].
    R_D : float
        Effective diode dynamic resistance for conduction model [Ω].
    V_0_D : float
        Effective diode forward knee voltage for conduction model [V].
    alpha_I : ndarray
        Discrete decay factors exp(-dt/τ) for IGBT junction→case Foster branches [-].
    alpha_D : ndarray
        Discrete decay factors exp(-dt/τ) for diode junction→case Foster branches [-].
    alpha_p : ndarray
        Discrete decay factor(s) for paste (case→sink) branch(es) [-].
    r_I : ndarray
        IGBT junction→case Foster branch thermal resistances [K/W].
    r_D : ndarray
        Diode junction→case Foster branch thermal resistances [K/W].
    r_paste : ndarray
        Paste (case→sink) thermal resistance(s) [K/W].
    r_sink : ndarray
        Heatsink (sink→ambient) Foster branch thermal resistances [K/W].
    Tamb : float
        Ambient temperature [K].

    Returns
    -------
    t : ndarray
        Time vector over the simulated 1 s window [s].
    m : ndarray
        Instantaneous modulation function m(t) [-].
    is_I : ndarray
        Instantaneous IGBT current (blocked when negative) [A].
    is_D : ndarray
        Instantaneous diode current (blocked when negative) [A].
    P_sw_I : ndarray
        IGBT switching loss power [W].
    P_sw_D : ndarray
        Diode switching loss power [W].
    P_con_I : ndarray
        IGBT conduction loss power [W].
    P_con_D : ndarray
        Diode conduction loss power [W].
    P_leg : ndarray
        Total leg loss P_I + P_D [W].
    T_j_I : ndarray
        IGBT junction temperature trajectory [K].
    T_j_D : ndarray
        Diode junction temperature trajectory [K].
    vs_inverter : ndarray
        Instantaneous AC-side phase voltage [V].
    is_inverter : ndarray
        Instantaneous AC-side phase current [A].

    """

    global Tbr_I, Tbr_D, Tbr_p, Tbr_s

    t = np.arange(0.0, 1.0, dt) # Create an array of time instants [s] (1000 steps) from 0 to 1.0 second. This defines the simulation horizon for that particular pf, P and Q value.

    vs_inverter, is_inverter = Inverter_voltage_and_current(Vs, Is, phi, t, omega)
    m = Instantaneous_modulation(M,omega,t,phi)
    is_I, is_D = IGBT_and_diode_current(Is, t, m)
    P_sw_I, P_sw_D = Switching_losses(V_dc, is_I, t_on, t_off, f_sw, is_D, I_ref, V_ref, Err_D)
    P_con_I, P_con_D = Conduction_losses(is_I, R_IGBT, V_0_IGBT, M, pf, is_D, R_D, V_0_D)


    P_I = np.maximum(P_sw_I + P_con_I, 0.0) # [W] Total instantaneous IGBT power loss [W] = switching + conduction
    P_D = np.maximum(P_sw_D + P_con_D, 0.0) # [W] Total instantaneous diode power loss [W] = switching + conduction
    P_leg = P_I + P_D                           # [W] Total leg power loss [W] = combined IGBT + diode losses

    # Initialize arrays for storing junction temperature profiles
    T_j_I = np.zeros_like(t, dtype=float)  # [K] IGBT junction temperature
    T_j_D = np.zeros_like(t, dtype=float)  # [K] Diode junction temperature

    for k in range(len(t)):

        # ----------------------------------------#
        # Junction-to-case (j→c) thermal RC update
        # ----------------------------------------#

        # Each Foster RC branch integrates the effect of the instantaneous semiconductor power loss (P_I or P_D) into a temperature rise.

        Tbr_I = alpha_I * Tbr_I + (1.0 - alpha_I) * (r_I * P_I[k]) # [K] IGBT
        Tbr_D = alpha_D * Tbr_D + (1.0 - alpha_D) * (r_D * P_D[k]) # [K] Diode

        # ----------------------------------------#
        # Case-to-sink and sink-to-ambient update
        # ----------------------------------------#

        # Shared thermal path (paste + heatsink) is driven by the total leg power
        Tbr_p = alpha_p * Tbr_p + (1.0 - alpha_p) * (r_paste * P_leg[k]) # [K] Thermal interface (paste)
        Tbr_s = alpha_s * Tbr_s + (1.0 - alpha_s) * (r_sink * P_leg[k])  # [K ]Heatsink branches

        shared_rise = Tbr_p.sum() + Tbr_s.sum()
        T_j_I[k] = Tamb + shared_rise + Tbr_I.sum()
        T_j_D[k] = Tamb + shared_rise + Tbr_D.sum()

    return (t,
            m,
            is_I,
            is_D ,
            P_sw_I,
            P_sw_D,
            P_con_I,
            P_con_D,
            P_leg,
            T_j_I,
            T_j_D,
            vs_inverter,
            is_inverter)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# main
#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

P_leg_all = []
TjI_all   = []
TjD_all   = []
all_rows = []

for sec_idx, (Vs_i, Is_i, phi_i, Vdc_i, pf_i) in enumerate(zip(Vs, Is, phi, V_dc, pf)):

    t, m, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D, P_leg, T_j_I, T_j_D, vs_inverter, is_inverter = mother_function(dt=dt,             # Input = Float
                                                                                              Vs=Vs_i,           # Input = Float, Originally = Array, Will go in loop
                                                                                              Is=Is_i,           # Input = Float, Originally = Array, Will go in loop
                                                                                              phi=phi_i,         # Input = Float, Originally = Array, Will go in loop
                                                                                              omega=omega,       # Input = Float
                                                                                              M=M,               # Input = Float
                                                                                              V_dc=Vdc_i,        # Input = Float, Originally = Array, Will go in loop
                                                                                              t_on=t_on,         # Input = Float
                                                                                              t_off=t_off,       # Input = Float
                                                                                              f_sw=f_sw,         # Input = Float
                                                                                              I_ref=I_ref,       # Input = Float
                                                                                              V_ref=V_ref,       # Input = Float
                                                                                              Err_D=Err_D,       # Input = Float
                                                                                              R_IGBT=R_IGBT,     # Input = Float
                                                                                              V_0_IGBT=V_0_IGBT, # Input = Float
                                                                                              pf=pf_i,           # Input = Float, Originally = Array, Will go in loop
                                                                                              R_D=R_D,           # Input = Float
                                                                                              V_0_D=V_0_D,       # Input = Float
                                                                                              alpha_I=alpha_I,   # Input = Array
                                                                                              alpha_D=alpha_D,   # Input = Array
                                                                                              alpha_p=alpha_p,   # Input = Array
                                                                                              r_I=r_I,           # Input = Array
                                                                                              r_D=r_D,           # Input = Array
                                                                                              r_paste=r_paste,   # Input = Array
                                                                                              r_sink=r_sink,     # Input = Array
                                                                                              Tamb=Tamb)         # Input = Float

    P_leg_all.append(P_leg)
    TjI_all.append(T_j_I)
    TjD_all.append(T_j_D)

    t_abs = t + sec_idx
    df_1 = pd.DataFrame({
        "sec_idx":sec_idx+1,
        "time_s": t_abs,
        "m": m,
        "is_I": is_I,
        "is_D": is_D,
        "P_sw_I": P_sw_I,
        "P_sw_D": P_sw_D,
        "P_con_I": P_con_I,
        "P_con_D": P_con_D,
        "P_leg": P_leg,
        "T_j_I": T_j_I,
        "T_j_D": T_j_D,
        "vs_inverter":vs_inverter,
        "is_inverter":is_inverter,
    })
    all_rows.append(df_1)

P_leg_all = np.concatenate(P_leg_all)  # [W] total leg power over full run
TjI_all   = np.concatenate(TjI_all)    # [K] IGBT junction temp over full run
TjD_all   = np.concatenate(TjD_all)    # [K] Diode junction temp over full run

ncycles = len(P_leg_all) // Ngrid                   # number of full cycles present
N = ncycles * Ngrid                                 # total samples that fit whole cycles
Pleg_cyc = P_leg_all[:N].reshape(ncycles, Ngrid)
TjI_cyc  = TjI_all[:N].reshape(ncycles, Ngrid)
TjD_cyc  = TjD_all[:N].reshape(ncycles, Ngrid)

# Means
TjI_mean  = TjI_cyc.mean(axis=1)  # [K] mean IGBT junction temperature per cycle
TjD_mean  = TjD_cyc.mean(axis=1)  # [K] mean Diode junction temperature per cycle
Pleg_mean = Pleg_cyc.mean(axis=1) # [W] mean total leg power per cycle

# Deltas
TjI_delta  = np.ptp(TjI_cyc, axis=1)   # [K] IGBT junction temperature ripple per cycle
TjD_delta  = np.ptp(TjD_cyc, axis=1)   # [K] Diode junction temperature ripple per cycle
Pleg_delta = np.ptp(Pleg_cyc, axis=1)  # [W] total leg power ripple per cycle



a_r_I, t_cycle_heat_I = ar_per_cycle(TjI_cyc, dt)
a_r_D, t_cycle_heat_D = ar_per_cycle(TjD_cyc, dt)

A     = 3.4368e14                    # [-]
alpha = -4.923                       # [-] 5 K ≤ ΔT_junc ≤ 80 K       # This condition will not be satisfied as junction temperature will never go this low. the author will ignore this condition.
beta1 = 9.012e-3                     # [-]
beta0 = 1.942                        # [-] 0.19 ≤ ar ≤ 0.42           # The ramp rate will be calculated and sometimes it will not satisfy this condition. Nevertheless the author will ignore this condition.
C     = 1.434                        # [-]
gamma = -1.208                       # [-] 0.07 s ≤ th ≤ 63 s         # As t_on
fd    = 0.6204                       # [-]
Ea    = 0.06606 * 1.60218e-19        # [J] 32.5 °C ≤ T_junc ≤ 122 °C  # This condition will be met most of the time but sometimes it will deviate. the author will ignore this condition.
k_b    = 8.6173324e-5 * 1.60218e-19  # [J/K]


def Cycles_to_failure(A,            # Input = float
                      alpha,        # Input = float
                      beta1,        # Input = float
                      beta0,        # Input = float
                      C,            # Input = float
                      gamma,        # Input = float
                      fd,           # Input = float
                      Ea,           # Input = float
                      k_b,          # Input = float
                      Tj_mean,      # Input = array
                      delta_Tj,     # Input = array
                      t_cycle_heat, # Input = array
                      ar ):         # Input = array

    """
    Compute cycles-to-failure Nf from the multi-parameter lifetime model.

    Parameters
    ----------
    A : float
        Empirical scale factor [-].
    alpha : float
        Exponent on temperature swing ΔTj [-].
    beta1 : float
        Slope vs. ΔTj in the a_r exponent term [-].
    beta0 : float
        Intercept in the a_r exponent term [-].
    C : float
        Time-shape factor [-].
    gamma : float
        Time-shape exponent [-].
    fd : float
        Damage factor (duty/usage factor) [-].
    Ea : float
        Activation energy [J].  (Note: pass Joules, not eV.)
    k_b : float
        Boltzmann constant [J/K].
    Tj_mean : array-like
        Mean junction temperature per cycle T̄j [K].
    delta_Tj : array-like
        Junction temperature swing per cycle ΔTj [K].
    t_cycle_heat : array-like
        Heating duration per cycle t_h [s] (time from Tmin→Tmax).
    ar : array-like
        Average temperature rise rate a_r per cycle [K/s] (= ΔTj / t_h).

    Returns
    -------
    Nf : ndarray
        Cycles-to-failure estimate for each cycle [-].
    """

    if np.any(Tj_mean <= 0):
        raise ValueError("Tj_mean contains 0 K or negative values, which is not physically possible. Check input data.")


    Equation_1 = A

    Equation_2 = (delta_Tj) ** alpha

    Equation_3 = (ar)** ((beta1*delta_Tj)+beta0)

    Equation_4 =  (C+((t_cycle_heat)**gamma))/(C+1)

    Equation_5 = np.exp(Ea/(k_b*Tj_mean))

    Equation_6 = fd

    Nf  = Equation_1 * Equation_2 * Equation_3 * Equation_4 * Equation_5 * Equation_6

    return Nf



Nf_I = Cycles_to_failure(A=A,
                      alpha=alpha,
                      beta1=beta1,
                      beta0=beta0,
                      C=C,
                      gamma=gamma,
                      fd=fd,
                      Ea=Ea,
                      k_b=k_b,
                      Tj_mean=TjI_mean,
                      delta_Tj=TjI_delta,
                      t_cycle_heat=t_cycle_heat_I,
                      ar=a_r_I )

Nf_D = Cycles_to_failure(A=A,
                      alpha=alpha,
                      beta1=beta1,
                      beta0=beta0,
                      C=C,
                      gamma=gamma,
                      fd=fd,
                      Ea=Ea,
                      k_b=k_b,
                      Tj_mean=TjD_mean,
                      delta_Tj=TjD_delta,
                      t_cycle_heat=t_cycle_heat_D,
                      ar=a_r_D )


df_2 = pd.DataFrame({
    "TjI_mean": TjI_mean,
    "TjD_mean": TjD_mean,
    "Pleg_mean": Pleg_mean,
    "TjI_delta": TjI_delta,
    "TjD_delta": TjD_delta,
    "Pleg_delta": Pleg_delta,
    "Nf_I": Nf_I,
    "Nf_D": Nf_D
})


df_1 = pd.concat(all_rows, ignore_index=True)
df_1.to_parquet("dataframe_files/mother_function_output_1.parquet", engine="pyarrow", compression="zstd")

df_2 = pd.concat(all_rows, ignore_index=True)
df_2.to_parquet("dataframe_files/mother_function_output_2.parquet", engine="pyarrow", compression="zstd")


def ttf_years_from_profile(Nf, f_hz, profile_seconds=None):
    """
    Time-to-failure (years) for a repeating mission profile using Miner’s rule.
    Nf: array of cycles-to-failure for each encountered cycle (same length as run)
    f_hz: cycle rate (e.g., 50 for 50 Hz)
    profile_seconds: duration of one pass of the profile; defaults to len(Nf)/f_hz
    """
    Nf = np.asarray(Nf, dtype=float)
    dmg_per_cycle = 1.0 / np.where(Nf > 0, Nf, np.inf)
    LC_one_pass = dmg_per_cycle.sum()
    if LC_one_pass <= 0:
        return np.inf
    if profile_seconds is None:
        profile_seconds = len(Nf) / float(f_hz)
    repeats_to_fail = 1.0 / LC_one_pass
    ttf_seconds = repeats_to_fail * profile_seconds
    return ttf_seconds / (365.0 * 24.0 * 3600.0)


TTF_years_IGBT  = ttf_years_from_profile(Nf_I, f_hz=f)
























