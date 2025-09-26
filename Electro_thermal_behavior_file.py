from Calculation_functions_file import Calculation_functions_class
import numpy as np

Calculation_functions = Calculation_functions_class()


class Electro_thermal_behavior_class:

    @staticmethod

    def Electro_thermal_behavior_shared_thermal_state(dt,       # Input = Float
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
                        alpha_s,  # Input = Array
                        r_I,      # Input = Array
                        r_D,      # Input = Array
                        r_paste,  # Input = Array
                        r_sink,   # Input = Array
                        Tamb,     # Input = Float
                        Tbr_I,    # Input = Array
                        Tbr_D,    # Input = Array
                        Tbr_p,    # Input = Array
                        Tbr_s):   # Input = Array

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

        t = np.arange(0.0, 1.0, dt)            # Create an array of time instants [s] (1000 steps) from 0 to 1.0 second. This defines the simulation horizon for that particular pf, P and Q value.

        vs_inverter, is_inverter = Calculation_functions.Inverter_voltage_and_current(Vs=Vs, Is=Is, phi=phi, t=t, omega=omega)
        m = Calculation_functions.Instantaneous_modulation(M=M,omega=omega,t=t,phi=phi)
        is_I, is_D = Calculation_functions.IGBT_and_diode_current(Is=Is, t=t, m=m,omega=omega)
        P_sw_I, P_sw_D = Calculation_functions.Switching_losses(V_dc=V_dc, is_I=is_I, t_on=t_on, t_off=t_off, f_sw=f_sw, is_D=is_D, I_ref=I_ref, V_ref=V_ref, Err_D=Err_D)
        P_con_I, P_con_D = Calculation_functions.Conduction_losses(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M, pf=pf, is_D=is_D, R_D=R_D, V_0_D=V_0_D)

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

            # IGBT junction→case branches driven by IGBT power P_I
            Tbr_I[:] = Tbr_I * alpha_I + (1.0 - alpha_I) * (r_I * P_I[k])

            # Diode junction→case branches driven by diode power P_D
            Tbr_D[:] = Tbr_D * alpha_D + (1.0 - alpha_D) * (r_D * P_D[k])

            # Paste (case→sink) driven by total leg power
            Tbr_p[:] = Tbr_p * alpha_p + (1.0 - alpha_p) * (r_paste * P_leg[k])

            # Heatsink branches (sink→ambient) also driven by total leg power
            Tbr_s[:] = Tbr_s * alpha_s + (1.0 - alpha_s) * (r_sink * P_leg[k])

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
                is_inverter,
                Tbr_I,
                Tbr_D,
                Tbr_p,
                Tbr_s)