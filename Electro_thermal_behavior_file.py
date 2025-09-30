import numpy as np
from numba import njit
from Calculation_functions_file import Calculation_functions_class

@njit(fastmath=True)
def thermal_rollout_shared_thermal_state(P_I,     # Input = ndarray
                                         P_D,     # Input = ndarray
                                         P_leg,   # Input = ndarray
                                         alpha_I, # Input = ndarray
                                         r_I,     # Input = ndarray
                                         alpha_D, # Input = ndarray
                                         r_D,     # Input = ndarray
                                         alpha_p, # Input = ndarray
                                         r_paste, # Input = ndarray
                                         alpha_s, # Input = ndarray
                                         r_sink,  # Input = ndarray
                                         Tamb,    # Input = float
                                         Tbr_I,   # Input/Output = ndarray
                                         Tbr_D,   # Input/Output = ndarray
                                         Tbr_p,   # Input/Output = ndarray
                                         Tbr_s):  # Input/Output = ndarray

    """

    Advance thermal RC branch states over one 1-second simulation window.

    This function is JIT-compiled with Numba for speed. It updates
    the Foster thermal network for IGBT, diode, paste, and heatsink
    branches given the instantaneous power losses.

    Parameters
    ----------
    P_I : ndarray
        Instantaneous IGBT power losses [W] (switching + conduction).
    P_D : ndarray
        Instantaneous diode power losses [W] (switching + conduction).
    P_leg : ndarray
        Instantaneous total leg power losses [W].
    alpha_I : ndarray
        Discrete decay factors exp(-dt/τ) for IGBT junction→case branches [-].
    r_I : ndarray
        Thermal resistances for IGBT junction→case branches [K/W].
    alpha_D : ndarray
        Discrete decay factors for diode junction→case branches [-].
    r_D : ndarray
        Thermal resistances for diode junction→case branches [K/W].
    alpha_p : ndarray
        Discrete decay factors for paste (case→sink) branches [-].
    r_paste : ndarray
        Thermal resistances for paste (case→sink) branches [K/W].
    alpha_s : ndarray
        Discrete decay factors for heatsink (sink→ambient) branches [-].
    r_sink : ndarray
        Thermal resistances for heatsink (sink→ambient) branches [K/W].
    Tamb : float
        Ambient temperature [K].
    Tbr_I : ndarray
        Initial Foster branch temperatures for IGBT junction→case [K].
        Updated in-place and returned.
    Tbr_D : ndarray
        Initial Foster branch temperatures for diode junction→case [K].
        Updated in-place and returned.
    Tbr_p : ndarray
        Initial temperature rise for paste (case→sink) [K].
        Updated in-place and returned.
    Tbr_s : ndarray
        Initial Foster branch temperatures for heatsink (sink→ambient) [K].
        Updated in-place and returned.

    Returns
    -------
    T_j_I : ndarray
        IGBT junction temperature trajectory over the 1 s window [K].
    T_j_D : ndarray
        Diode junction temperature trajectory over the 1 s window [K].
    Tbr_I : ndarray
        Final Foster branch temperatures for IGBT junction→case [K].
    Tbr_D : ndarray
        Final Foster branch temperatures for diode junction→case [K].
    Tbr_p : ndarray
        Final paste (case→sink) temperature rise [K].
    Tbr_s : ndarray
        Final Foster branch temperatures for heatsink (sink→ambient) [K].

    """

    n = P_I.size

    T_j_I = np.empty(n, dtype=np.float64)
    T_j_D = np.empty(n, dtype=np.float64)

    # Precompute per-branch gains k = (1 - alpha) * r
    kI = (1.0 - alpha_I) * r_I
    kD = (1.0 - alpha_D) * r_D
    kp = (1.0 - alpha_p) * r_paste
    ks = (1.0 - alpha_s) * r_sink

    for k in range(n):
        # IGBT junction→case branches (driven by P_I[k])
        for i in range(alpha_I.size):
            Tbr_I[i] = Tbr_I[i] * alpha_I[i] + kI[i] * P_I[k]

        # Diode junction→case branches (driven by P_D[k])
        for i in range(alpha_D.size):
            Tbr_D[i] = Tbr_D[i] * alpha_D[i] + kD[i] * P_D[k]

        # Paste (case→sink) branches (driven by total leg power)
        for i in range(alpha_p.size):
            Tbr_p[i] = Tbr_p[i] * alpha_p[i] + kp[i] * P_leg[k]

        # Heatsink (sink→ambient) branches (driven by total leg power)
        for i in range(alpha_s.size):
            Tbr_s[i] = Tbr_s[i] * alpha_s[i] + ks[i] * P_leg[k]

        # Sums for junction temperatures
        shared_rise = 0.0
        for i in range(alpha_p.size):
            shared_rise += Tbr_p[i]
        for i in range(alpha_s.size):
            shared_rise += Tbr_s[i]

        sum_I = 0.0
        for i in range(alpha_I.size):
            sum_I += Tbr_I[i]
        sum_D = 0.0
        for i in range(alpha_D.size):
            sum_D += Tbr_D[i]

        T_j_I[k] = Tamb + shared_rise + sum_I
        T_j_D[k] = Tamb + shared_rise + sum_D

    return T_j_I, T_j_D, Tbr_I, Tbr_D, Tbr_p, Tbr_s




@njit(fastmath=True)
def thermal_rollout_separated_thermal_state(P_I,        # ndarray
                                            P_D,        # ndarray
                                            alpha_I,    # ndarray
                                            r_I,        # ndarray
                                            alpha_D,    # ndarray
                                            r_D,        # ndarray
                                            alpha_p,    # ndarray
                                            r_paste,    # ndarray
                                            alpha_s,    # ndarray
                                            r_sink,     # ndarray
                                            Tamb,       # float
                                            Tbr_I,      # ndarray (in/out) IGBT junction→case
                                            Tbr_D,      # ndarray (in/out) Diode junction→case
                                            Tbr_p_I,    # ndarray (in/out) IGBT paste (case→sink)
                                            Tbr_s_I,    # ndarray (in/out) IGBT sink  (sink→ambient)
                                            Tbr_p_D,    # ndarray (in/out) Diode paste (case→sink)
                                            Tbr_s_D):     # ndarray (in/out) Diode sink  (sink→ambient)

    """
    Advance thermal RC branch states over one 1-second simulation window,
    with *independent* paste and heatsink thermal paths for IGBT and Diode.

    This function is JIT-compiled with Numba for speed. It updates
    the Foster thermal network for each device (IGBT, diode) separately:
        - Junction → case
        - Case → paste → sink → ambient

    Parameters
    ----------
    P_I : ndarray
        Instantaneous IGBT power losses [W] (switching + conduction).
    P_D : ndarray
        Instantaneous diode power losses [W] (switching + conduction).
    alpha_I : ndarray
        Discrete decay factors exp(-dt/τ) for IGBT junction→case branches [-].
    r_I : ndarray
        Thermal resistances for IGBT junction→case branches [K/W].
    alpha_D : ndarray
        Discrete decay factors for diode junction→case branches [-].
    r_D : ndarray
        Thermal resistances for diode junction→case branches [K/W].
    alpha_p : ndarray
        Discrete decay factors for paste (case→sink) branches [-].
    r_paste : ndarray
        Thermal resistances for paste (case→sink) branches [K/W].
    alpha_s : ndarray
        Discrete decay factors for heatsink (sink→ambient) branches [-].
    r_sink : ndarray
        Thermal resistances for heatsink (sink→ambient) branches [K/W].
    Tamb : float
        Ambient temperature [K].
    Tbr_I : ndarray
        Foster branch states for IGBT junction→case [K].
        Updated in-place and returned.
    Tbr_D : ndarray
        Foster branch states for diode junction→case [K].
        Updated in-place and returned.
    Tbr_p_I : ndarray
        Foster branch states for IGBT paste (case→sink) [K].
        Updated in-place and returned.
    Tbr_s_I : ndarray
        Foster branch states for IGBT heatsink (sink→ambient) [K].
        Updated in-place and returned.
    Tbr_p_D : ndarray
        Foster branch states for diode paste (case→sink) [K].
        Updated in-place and returned.
    Tbr_s_D : ndarray
        Foster branch states for diode heatsink (sink→ambient) [K].
        Updated in-place and returned.

    Returns
    -------
    T_j_I : ndarray
        IGBT junction temperature trajectory over the 1 s window [K].
    T_j_D : ndarray
        Diode junction temperature trajectory over the 1 s window [K].
    Tbr_I : ndarray
        Final Foster branch states for IGBT junction→case [K].
    Tbr_D : ndarray
        Final Foster branch states for diode junction→case [K].
    Tbr_p_I : ndarray
        Final paste states for IGBT case→sink [K].
    Tbr_s_I : ndarray
        Final heatsink states for IGBT sink→ambient [K].
    Tbr_p_D : ndarray
        Final paste states for diode case→sink [K].
    Tbr_s_D : ndarray
        Final heatsink states for diode sink→ambient [K].
    """

    n = P_I.size

    T_j_I = np.empty(n, dtype=np.float64)
    T_j_D = np.empty(n, dtype=np.float64)

    # Precompute per-branch gains k = (1 - alpha) * r
    kI = (1.0 - alpha_I) * r_I
    kD = (1.0 - alpha_D) * r_D
    kp = (1.0 - alpha_p) * r_paste
    ks = (1.0 - alpha_s) * r_sink

    for k in range(n):
        # IGBT junction→case branches
        for i in range(alpha_I.size):
            Tbr_I[i] = Tbr_I[i] * alpha_I[i] + kI[i] * P_I[k]

        # Diode junction→case branches
        for i in range(alpha_D.size):
            Tbr_D[i] = Tbr_D[i] * alpha_D[i] + kD[i] * P_D[k]

        # IGBT paste (case→sink) driven by IGBT loss
        for i in range(alpha_p.size):
            Tbr_p_I[i] = Tbr_p_I[i] * alpha_p[i] + kp[i] * P_I[k]

        # IGBT sink (sink→ambient) driven by IGBT loss
        for i in range(alpha_s.size):
            Tbr_s_I[i] = Tbr_s_I[i] * alpha_s[i] + ks[i] * P_I[k]

        # Diode paste (case→sink) driven by Diode loss
        for i in range(alpha_p.size):
            Tbr_p_D[i] = Tbr_p_D[i] * alpha_p[i] + kp[i] * P_D[k]

        # Diode sink (sink→ambient) driven by Diode loss
        for i in range(alpha_s.size):
            Tbr_s_D[i] = Tbr_s_D[i] * alpha_s[i] + ks[i] * P_D[k]

        # Sums for junction temperatures (independent paths)
        sum_I = 0.0
        for i in range(alpha_I.size):
            sum_I += Tbr_I[i]
        sum_D = 0.0
        for i in range(alpha_D.size):
            sum_D += Tbr_D[i]

        rise_p_I = 0.0
        for i in range(alpha_p.size):
            rise_p_I += Tbr_p_I[i]
        rise_s_I = 0.0
        for i in range(alpha_s.size):
            rise_s_I += Tbr_s_I[i]

        rise_p_D = 0.0
        for i in range(alpha_p.size):
            rise_p_D += Tbr_p_D[i]
        rise_s_D = 0.0
        for i in range(alpha_s.size):
            rise_s_D += Tbr_s_D[i]

        T_j_I[k] = Tamb + rise_p_I + rise_s_I + sum_I
        T_j_D[k] = Tamb + rise_p_D + rise_s_D + sum_D

    return T_j_I, T_j_D, Tbr_I, Tbr_D, Tbr_p_I, Tbr_s_I, Tbr_p_D, Tbr_s_D



class Electro_thermal_behavior_class:

    @staticmethod
    def Electro_thermal_behavior_shared_thermal_state(dt,        # float
                                                      Vs,        # float
                                                      Is,        # float
                                                      phi,       # float
                                                      omega,     # float
                                                      M,         # float
                                                      V_dc,      # float
                                                      t_on,      # float
                                                      t_off,     # float
                                                      f_sw,      # float
                                                      I_ref,     # float
                                                      V_ref,     # float
                                                      Err_D,     # float
                                                      R_IGBT,    # float
                                                      V_0_IGBT,  # float
                                                      pf,        # float
                                                      R_D,       # float
                                                      V_0_D,     # float
                                                      alpha_I,   # ndarray
                                                      alpha_D,   # ndarray
                                                      alpha_p,   # ndarray
                                                      alpha_s,   # ndarray
                                                      r_I,       # ndarray
                                                      r_D,       # ndarray
                                                      r_paste,   # ndarray
                                                      r_sink,    # ndarray
                                                      Tamb,      # float
                                                      Tbr_I,     # ndarray
                                                      Tbr_D,     # ndarray
                                                      Tbr_p,     # ndarray
                                                      Tbr_s ):  # ndarray

        """
        
        Simulate one-second electro-thermal behavior of a single inverter leg and return
        electrical waveforms, loss components and device junction temperatures.

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
        Tbr_I : ndarray
            Temperature rise contributions of each Foster RC branch for IGBT junction → case. [K].
        Tbr_D : ndarray
            Temperature rise contributions of each Foster RC branch for diode junction → case. [K].
        Tbr_p : ndarray
            Temperature rise across the thermal paste (case → heatsink interface). [K].
        Tbr_s : ndarray
            Temperature rise contributions of each Foster RC branch for the heatsink (sink → ambient). [K].

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
        Tbr_I : ndarray
            Temperature rise contributions of each Foster RC branch for IGBT junction → case. [K].
        Tbr_D : ndarray
            Temperature rise contributions of each Foster RC branch for diode junction → case. [K].
        Tbr_p : ndarray
            Temperature rise across the thermal paste (case → heatsink interface). [K].
        Tbr_s : ndarray
            Temperature rise contributions of each Foster RC branch for the heatsink (sink → ambient). [K].

        Note
        -------
        Tbr_I, Tbr_D, Tbr_p, and Tbr_s all start from zero at each time step, with every new step beginning from the final temperature conditions of the previous one.

        """

        # time vector for this 1-second window
        t = np.arange(0.0, 1.0, dt, dtype=np.float64)  # Create an array of time instants [s] (1000 steps) from 0 to 1.0 second. This defines the simulation horizon for that particular pf, P and Q value.

        vs_inverter, is_inverter = Calculation_functions_class().Inverter_voltage_and_current(Vs=Vs, Is=Is, phi=phi, t=t, omega=omega)
        m                        = Calculation_functions_class().Instantaneous_modulation(M=M,omega=omega,t=t,phi=phi)
        is_I, is_D               = Calculation_functions_class().IGBT_and_diode_current(Is=Is, t=t, m=m,omega=omega)
        P_sw_I, P_sw_D           = Calculation_functions_class().Switching_losses(V_dc=V_dc, is_I=is_I, t_on=t_on, t_off=t_off, f_sw=f_sw, is_D=is_D, I_ref=I_ref, V_ref=V_ref, Err_D=Err_D)
        P_con_I, P_con_D         = Calculation_functions_class().Conduction_losses(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M, pf=pf, is_D=is_D, R_D=R_D, V_0_D=V_0_D)

        P_I = np.maximum(P_sw_I + P_con_I, 0.0)   # [W] Total instantaneous IGBT power loss [W] = switching + conduction
        P_D = np.maximum(P_sw_D + P_con_D, 0.0)   # [W] Total instantaneous diode power loss [W] = switching + conduction
        P_leg = P_I + P_D                             # [W] Total leg power loss [W] = combined IGBT + diode losses

        # Ensure all arrays for the JIT kernel are float64 & contiguous
        alpha_I = np.ascontiguousarray(alpha_I, dtype=np.float64)
        alpha_D = np.ascontiguousarray(alpha_D, dtype=np.float64)
        alpha_p = np.ascontiguousarray(alpha_p, dtype=np.float64)
        alpha_s = np.ascontiguousarray(alpha_s, dtype=np.float64)

        r_I     = np.ascontiguousarray(r_I,     dtype=np.float64)
        r_D     = np.ascontiguousarray(r_D,     dtype=np.float64)
        r_paste = np.ascontiguousarray(r_paste, dtype=np.float64)
        r_sink  = np.ascontiguousarray(r_sink,  dtype=np.float64)

        Tbr_I   = np.ascontiguousarray(Tbr_I,   dtype=np.float64)
        Tbr_D   = np.ascontiguousarray(Tbr_D,   dtype=np.float64)
        Tbr_p   = np.ascontiguousarray(Tbr_p,   dtype=np.float64)
        Tbr_s   = np.ascontiguousarray(Tbr_s,   dtype=np.float64)

        P_I   = np.ascontiguousarray(P_I,   dtype=np.float64)
        P_D   = np.ascontiguousarray(P_D,   dtype=np.float64)
        P_leg = np.ascontiguousarray(P_leg, dtype=np.float64)

        # Advance thermal RC states over this 1-second window in compiled code
        T_j_I, T_j_D, Tbr_I, Tbr_D, Tbr_p, Tbr_s = thermal_rollout_shared_thermal_state(P_I=P_I,
                                                                                        P_D=P_D,
                                                                                        P_leg=P_leg,
                                                                                        alpha_I=alpha_I,
                                                                                        r_I=r_I,
                                                                                        alpha_D=alpha_D,
                                                                                        r_D=r_D,
                                                                                        alpha_p=alpha_p,
                                                                                        r_paste=r_paste,
                                                                                        alpha_s=alpha_s,
                                                                                        r_sink=r_sink,
                                                                                        Tamb = float(Tamb),
                                                                                        Tbr_I = Tbr_I.copy(),
                                                                                        Tbr_D = Tbr_D.copy(),
                                                                                        Tbr_p = Tbr_p.copy(),
                                                                                        Tbr_s = Tbr_s.copy())

        return (t,
                m,
                is_I,
                is_D,
                P_sw_I,
                P_sw_D,
                P_con_I,
                P_con_D,
                P_leg,
                T_j_I,
                T_j_D,
                vs_inverter,
                is_inverter,
                Tbr_I,
                Tbr_D,
                Tbr_p,
                Tbr_s )



    @staticmethod
    def Electro_thermal_behavior_separated_thermal_state(dt,        # float
                                                         Vs,        # float
                                                         Is,        # float
                                                         phi,       # float
                                                         omega,     # float
                                                         M,         # float
                                                         V_dc,      # float
                                                         t_on,      # float
                                                         t_off,     # float
                                                         f_sw,      # float
                                                         I_ref,     # float
                                                         V_ref,     # float
                                                         Err_D,     # float
                                                         R_IGBT,    # float
                                                         V_0_IGBT,  # float
                                                         pf,        # float
                                                         R_D,       # float
                                                         V_0_D,     # float
                                                         alpha_I,   # ndarray
                                                         alpha_D,   # ndarray
                                                         alpha_p,   # ndarray
                                                         alpha_s,   # ndarray
                                                         r_I,       # ndarray
                                                         r_D,       # ndarray
                                                         r_paste,   # ndarray
                                                         r_sink,    # ndarray
                                                         Tamb,      # float
                                                         Tbr_I,     # ndarray
                                                         Tbr_D,     # ndarray
                                                         Tbr_p_I,   # ndarray
                                                         Tbr_s_I,   # ndarray
                                                         Tbr_p_D,   # ndarray
                                                         Tbr_s_D ): # ndarray

        """
        Simulate one-second electro-thermal behavior of a single inverter leg and return
        electrical waveforms, loss components and device junction temperatures.

        This version assumes *separate* thermal paths for IGBT and diode.
        Each device has its own paste (case→sink) and heatsink (sink→ambient)
        RC networks, unlike the shared model.

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
        alpha_s : ndarray
            Discrete decay factor(s) for heatsink (sink→ambient) branch(es) [-].
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
        Tbr_I : ndarray
            Foster branch states for IGBT junction → case [K].
        Tbr_D : ndarray
            Foster branch states for diode junction → case [K].
        Tbr_p_I : ndarray
            Foster branch states for IGBT paste (case → sink) [K].
        Tbr_s_I : ndarray
            Foster branch states for IGBT heatsink (sink → ambient) [K].
        Tbr_p_D : ndarray
            Foster branch states for diode paste (case → sink) [K].
        Tbr_s_D : ndarray
            Foster branch states for diode heatsink (sink → ambient) [K].

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
        Tbr_I : ndarray
            Updated Foster branch states for IGBT junction → case [K].
        Tbr_D : ndarray
            Updated Foster branch states for diode junction → case [K].
        Tbr_p_I : ndarray
            Updated Foster branch states for IGBT paste (case → sink) [K].
        Tbr_s_I : ndarray
            Updated Foster branch states for IGBT heatsink (sink → ambient) [K].
        Tbr_p_D : ndarray
            Updated Foster branch states for diode paste (case → sink) [K].
        Tbr_s_D : ndarray
            Updated Foster branch states for diode heatsink (sink → ambient) [K].
        """

        # time vector for this 1-second window
        t = np.arange(0.0, 1.0, dt,dtype=np.float64)  # Create an array of time instants [s] (1000 steps) from 0 to 1.0 second. This defines the simulation horizon for that particular pf, P and Q value.

        vs_inverter, is_inverter = Calculation_functions_class().Inverter_voltage_and_current(Vs=Vs, Is=Is, phi=phi,t=t, omega=omega)
        m = Calculation_functions_class().Instantaneous_modulation(M=M, omega=omega, t=t, phi=phi)
        is_I, is_D = Calculation_functions_class().IGBT_and_diode_current(Is=Is, t=t, m=m, omega=omega)
        P_sw_I, P_sw_D = Calculation_functions_class().Switching_losses(V_dc=V_dc, is_I=is_I, t_on=t_on, t_off=t_off,f_sw=f_sw, is_D=is_D, I_ref=I_ref, V_ref=V_ref,Err_D=Err_D)
        P_con_I, P_con_D = Calculation_functions_class().Conduction_losses(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT,M=M, pf=pf, is_D=is_D, R_D=R_D, V_0_D=V_0_D)

        P_I = np.maximum(P_sw_I + P_con_I, 0.0)  # [W] Total instantaneous IGBT power loss [W] = switching + conduction
        P_D = np.maximum(P_sw_D + P_con_D, 0.0)  # [W] Total instantaneous diode power loss [W] = switching + conduction
        P_leg = P_I + P_D  # [W] Total leg power loss [W] = combined IGBT + diode losses

        # Ensure all arrays for the JIT kernel are float64 & contiguous
        alpha_I = np.ascontiguousarray(alpha_I, dtype=np.float64)
        alpha_D = np.ascontiguousarray(alpha_D, dtype=np.float64)
        alpha_p = np.ascontiguousarray(alpha_p, dtype=np.float64)
        alpha_s = np.ascontiguousarray(alpha_s, dtype=np.float64)

        r_I = np.ascontiguousarray(r_I, dtype=np.float64)
        r_D = np.ascontiguousarray(r_D, dtype=np.float64)
        r_paste = np.ascontiguousarray(r_paste, dtype=np.float64)
        r_sink = np.ascontiguousarray(r_sink, dtype=np.float64)

        Tbr_I = np.ascontiguousarray(Tbr_I, dtype=np.float64)
        Tbr_D = np.ascontiguousarray(Tbr_D, dtype=np.float64)
        Tbr_p_I = np.ascontiguousarray(Tbr_p_I, dtype=np.float64)
        Tbr_s_I = np.ascontiguousarray(Tbr_s_I, dtype=np.float64)
        Tbr_p_D = np.ascontiguousarray(Tbr_p_D, dtype=np.float64)
        Tbr_s_D = np.ascontiguousarray(Tbr_s_D, dtype=np.float64)

        P_I = np.ascontiguousarray(P_I, dtype=np.float64)
        P_D = np.ascontiguousarray(P_D, dtype=np.float64)
        P_leg = np.ascontiguousarray(P_D, dtype=np.float64)

        # Advance thermal RC states with separated paths
        T_j_I, T_j_D, Tbr_I, Tbr_D, Tbr_p_I, Tbr_s_I, Tbr_p_D, Tbr_s_D = thermal_rollout_separated_thermal_state(P_I=P_I,
                                                                                                                 P_D=P_D,
                                                                                                                 alpha_I=alpha_I,
                                                                                                                 r_I=r_I,
                                                                                                                 alpha_D=alpha_D,
                                                                                                                 r_D=r_D,
                                                                                                                 alpha_p=alpha_p,
                                                                                                                 r_paste=r_paste,
                                                                                                                 alpha_s=alpha_s,
                                                                                                                 r_sink=r_sink,
                                                                                                                 Tamb=float(Tamb),
                                                                                                                 Tbr_I=Tbr_I.copy(),
                                                                                                                 Tbr_D=Tbr_D.copy(),
                                                                                                                 Tbr_p_I=Tbr_p_I.copy(),
                                                                                                                 Tbr_s_I=Tbr_s_I.copy(),
                                                                                                                 Tbr_p_D=Tbr_p_D.copy(),
                                                                                                                 Tbr_s_D=Tbr_s_D.copy())

        return (t,
                m,
                is_I,
                is_D,
                P_sw_I,
                P_sw_D,
                P_con_I,
                P_con_D,
                P_leg,
                T_j_I,
                T_j_D,
                vs_inverter,
                is_inverter,
                Tbr_I,
                Tbr_D,
                Tbr_p_I,
                Tbr_s_I,
                Tbr_p_D,
                Tbr_s_D )

