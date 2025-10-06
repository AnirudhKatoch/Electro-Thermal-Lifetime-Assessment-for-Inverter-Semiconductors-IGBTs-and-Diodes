import numpy as np
from scipy.special import lambertw
import numexpr as ne
import gc, ctypes, sys
import os, re, glob
import pyarrow.parquet as pq
import pandas as pd

class Calculation_functions_class:

    @staticmethod
    def compute_power_flow_from_pf(P,
                                   Q,
                                   V_dc,
                                   pf,
                                   Vs,
                                   M,
                                   single_phase_inverter_topology,
                                   inverter_phases,
                                   modulation_scheme):
        """
        Compute apparent power S, RMS current Is, and phase angle phi

        Parameters
        ----------
        P : array
            Active power per sec [W].
        Q : array
            Reactive power per sec [VAr].
        V_dc : array
             DC-side phase voltage per sec [V]
        pf : array
             Power factor per sec [-].
        Vs : array
             RMS AC-side phase voltage per sec [V]
        single_phase_inverter_topology : {"half","full"}
            Inverter topology (affects Vs limit for single-phase).
        inverter_phases : {1,3}
            Number of phases. If 3, Vs is interpreted as PHASE RMS (i.e., V_ll/sqrt(3)).
        modulation_scheme : {"spwm","svm"}
            Modulation strategy used for generating inverter switching signals.


        Returns
        -------
        S : array
            Apparent power per sample [VA].
        Is : array
            RMS current per sample [A].
        phi : array
            Phase angle between voltage and current per sample [rad]
        P : ndarray of float
            Active power after applying pf==0 rule (P set to 0 where pf == 0) [W].
        Q : ndarray of float
            Reactive power after consistency update (recomputed where pf ≠ 0) [VAr].
        Vs : array
             RMS AC-side phase voltage per sec [V]
        """

        S = np.zeros_like(pf, dtype=float)  # [VA] Inverter RMS apparent power
        Is = np.zeros_like(pf, dtype=float)  # [A] Inverter RMS current
        phi = np.zeros_like(pf, dtype=float)  # [rad] Phase angle

        if inverter_phases == 1:
            if single_phase_inverter_topology == "full":
                Vs_theoretical = (M * V_dc) / np.sqrt(2.0)
            elif single_phase_inverter_topology == "half":
                Vs_theoretical = (M * V_dc) / (2.0 * np.sqrt(2.0))
        elif inverter_phases ==3:
            if modulation_scheme == "svm":
                # Space vector PWM (or 3rd harmonic injection)
                Vs_theoretical = (M * V_dc) / np.sqrt(6.0)  # [V RMS phase]
            elif modulation_scheme == "spwm" :  # "spwm"
                # Sinusoidal PWM
                Vs_theoretical = (M * V_dc) / (2.0 * np.sqrt(2.0))

        if Vs.size == 0:
            Vs = Vs_theoretical.copy()

        else:
            indices = np.where(Vs > Vs_theoretical)[0]
            if indices.size > 0:
                raise ValueError(
                    f"Invalid input: AC phase RMS voltage exceeds the theoretical limit "
                    f"Vs must not be greater than {np.max(Vs_theoretical)}."
                )

        '''
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
        '''

        # masks
        m0 = pf == 0  # zero power factor
        mneg = pf < 0  # inductive
        mpos = pf > 0  # capacitive

        # ---- pf == 0 branch ----
        # P[i] = 0
        P[m0] = 0.0

        # S[i] = sqrt(P[i]^2 + Q[i]^2)  (with P already zeroed where m0)
        S[m0] = np.sqrt(P[m0] ** 2 + Q[m0] ** 2)

        # Is[i] = S[i] / Vs[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            Is[m0] = S[m0] / (Vs[m0] if inverter_phases == 1 else (3.0 * Vs[m0]))

        # phi: 0 if S==0 else ±pi/2 depending on sign of Q
        phi[m0] = 0.0
        nz = m0 & (S != 0)
        phi[nz] = np.where(Q[nz] > 0, np.pi / 2, -np.pi / 2)

        # ---- pf != 0 branch ----
        abspf = np.abs(pf)
        mnz = ~m0  # pf != 0

        # S[i] = P[i] / abs(pf[i])
        S[mnz] = P[mnz] / abs(pf[mnz])

        # Is[i] = S[i] / Vs[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            Is[mnz] = S[mnz] / (Vs[mnz] if inverter_phases == 1 else (3.0 * Vs[mnz]))

        # phi[i] = ± arccos(abs(pf[i]))
        phi[mneg] = -np.arccos(abspf[mneg])  # inductive
        phi[mpos] = np.arccos(abspf[mpos])  # capacitive

        # Q[i] = ± sqrt(S[i]^2 - P[i]^2) for pf != 0
        # (Note: numerical noise can make the radicand slightly negative; clip at 0.)
        rad = np.clip(S[mnz] ** 2 - P[mnz] ** 2)
        root = np.sqrt(rad)
        idx_mnz = np.where(mnz)[0]
        Q[idx_mnz[mneg[mnz]]] = -root[mneg[mnz]]
        Q[idx_mnz[mpos[mnz]]] = root[mpos[mnz]]

        return S, Is, phi, P, Q, Vs

    @staticmethod
    def compute_power_flow_from_pf_design_control_inverter(overshoot_margin_inverter,
                                                           inverter_phases,
                                                           Vs,
                                                           max_IGBT_RMS_Current,
                                                           S,
                                                           P,
                                                           Q,
                                                           pf,
                                                           single_phase_inverter_topology,
                                                           modulation_scheme,
                                                           M,
                                                           V_dc):

        @staticmethod
        def compute_power_flow_from_pf_design_control_inverter(
                overshoot_margin_inverter,
                inverter_phases,
                Vs,
                max_IGBT_RMS_Current,
                S,
                P,
                Q,
                pf,
                single_phase_inverter_topology,
                modulation_scheme,
                M,
                V_dc
        ):
            """
            Compute apparent power S, RMS current Is, and phase angle phi
            with automatic parallel switch sizing (design-control version).

            Parameters
            ----------
            overshoot_margin_inverter : float
                Safety margin factor (e.g., 0.1 → 10%).
            inverter_phases : {1,3}
                Number of inverter phases.
            Vs : array
                RMS AC-side phase voltage per sec [V].
            max_IGBT_RMS_Current : float
                Maximum RMS current rating of one IGBT device [A].
            S : array
                Apparent power per sec [VA] (system level before scaling).
            P : array
                Active power per sec [W].
            Q : array
                Reactive power per sec [VAr].
            pf : array
                Power factor per sec [-].
            single_phase_inverter_topology : {"half","full"}
                Topology for single-phase inverter.
            modulation_scheme : {"spwm","svm"}
                Modulation strategy.
            M : float
                Modulation index [-].
            V_dc : array
                DC-side phase voltage per sec [V].

            Returns
            -------
            S : array
                Apparent power per sample [VA] (per-device after scaling).
            Is : array
                RMS current per sample [A] (per-device).
            phi : array
                Phase angle between voltage and current per sample [rad].
            P : array
                Active power per sample [W] (per-device after pf==0 rule).
            Q : array
                Reactive power per sample [VAr] (per-device, updated for pf ≠ 0).
            Vs : array
                RMS AC-side phase voltage per sec [V].
            N_parallel : int
                Number of parallel switches required.
            """

        margin = 1.0 + float(overshoot_margin_inverter)
        S_max_eff = inverter_phases * Vs * float(max_IGBT_RMS_Current) / margin
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(S_max_eff > 0, S / S_max_eff, 0.0)
        N_parallel = int(np.ceil(np.nanmax(ratio)))

        if N_parallel < 1:
            N_parallel = 1
        P = P / N_parallel
        Q = Q / N_parallel

        S = np.zeros_like(pf, dtype=float)  # [VA] Inverter RMS apparent power
        Is = np.zeros_like(pf, dtype=float)  # [A] Inverter RMS current
        phi = np.zeros_like(pf, dtype=float)  # [rad] Phase angle

        if inverter_phases == 1:
            if single_phase_inverter_topology == "full":
                Vs_theoretical = (M * V_dc) / np.sqrt(2.0)
            elif single_phase_inverter_topology == "half":
                Vs_theoretical = (M * V_dc) / (2.0 * np.sqrt(2.0))
        elif inverter_phases == 3:
            if modulation_scheme == "svm":
                # Space vector PWM (or 3rd harmonic injection)
                Vs_theoretical = (M * V_dc) / np.sqrt(6.0)  # [V RMS phase]
            elif modulation_scheme == "spwm":  # "spwm"
                # Sinusoidal PWM
                Vs_theoretical = (M * V_dc) / (2.0 * np.sqrt(2.0))

        if Vs.size == 0:
            Vs = Vs_theoretical.copy()

        else:
            indices = np.where(Vs > Vs_theoretical)[0]
            if indices.size > 0:
                raise ValueError(
                    f"Invalid input: AC phase RMS voltage exceeds the theoretical limit "
                    f"Vs must not be greater than {np.max(Vs_theoretical)}."
                )

        # masks
        m0 = pf == 0  # zero power factor
        mneg = pf < 0  # inductive
        mpos = pf > 0  # capacitive

        # ---- pf == 0 branch ----
        # P[i] = 0
        P[m0] = 0.0

        # S[i] = sqrt(P[i]^2 + Q[i]^2)  (with P already zeroed where m0)
        S[m0] = np.sqrt(P[m0] ** 2 + Q[m0] ** 2)

        # Is[i] = S[i] / Vs[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            Is[m0] = S[m0] / (Vs[m0] if inverter_phases == 1 else (3.0 * Vs[m0]))

        # phi: 0 if S==0 else ±pi/2 depending on sign of Q
        phi[m0] = 0.0
        nz = m0 & (S != 0)
        phi[nz] = np.where(Q[nz] > 0, np.pi / 2, -np.pi / 2)

        # ---- pf != 0 branch ----
        abspf = np.abs(pf)
        mnz = ~m0  # pf != 0

        # S[i] = P[i] / abs(pf[i])
        S[mnz] = P[mnz] / abs(pf[mnz])

        # Is[i] = S[i] / Vs[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            Is[mnz] = S[mnz] / (Vs[mnz] if inverter_phases == 1 else (3.0 * Vs[mnz]))

        # phi[i] = ± arccos(abs(pf[i]))
        phi[mneg] = -np.arccos(abspf[mneg])  # inductive
        phi[mpos] = np.arccos(abspf[mpos])  # capacitive

        # Q[i] = ± sqrt(S[i]^2 - P[i]^2) for pf != 0
        # (Note: numerical noise can make the radicand slightly negative; clip at 0.)
        rad = np.clip(S[mnz] ** 2 - P[mnz] ** 2)
        root = np.sqrt(rad)
        idx_mnz = np.where(mnz)[0]
        Q[idx_mnz[mneg[mnz]]] = -root[mneg[mnz]]
        Q[idx_mnz[mpos[mnz]]] = root[mpos[mnz]]

        return S, Is, phi, P, Q, Vs,N_parallel

    @staticmethod
    def Inverter_voltage_and_current(Vs, Is, phi, t, omega):
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

        #vs_inverter = np.sqrt(2) * Vs * np.sin(omega * t + phi)
        #is_inverter = np.sqrt(2) * Is * np.sin(omega * t)

        sqrt2 = np.sqrt(2.0)
        vs_inverter = ne.evaluate("sqrt2 * Vs * sin(omega * t + phi)",
                                  local_dict=dict(sqrt2=sqrt2,Vs=Vs,omega=omega,t=t,phi=phi),)
        is_inverter = ne.evaluate("sqrt2 * Is * sin(omega * t)",
                                  local_dict=dict(sqrt2=sqrt2,Is=Is,omega=omega,t=t),)

        return vs_inverter, is_inverter

    @staticmethod
    def Instantaneous_modulation(M, omega, t, phi):

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

    @staticmethod
    def IGBT_and_diode_current(Is, t, m, omega):

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

    @staticmethod
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

        is_I = np.ascontiguousarray(is_I, dtype=np.float64)
        is_D = np.ascontiguousarray(is_D, dtype=np.float64)

        # IGBT

        #E_on_I = ((np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_on)
        #E_off_I = ((np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_off)

        c1 = (np.sqrt(2) / (2 * np.pi))
        c2 = (np.sqrt(2) / np.pi)

        P_sw_I_expr = ne.evaluate("(( c1 * V_dc * is_I * t_on) + ( c1 * V_dc * is_I * t_off)) * f_sw",
                             local_dict=dict(c1=c1,V_dc=V_dc,is_I=is_I,t_on=t_on,t_off=t_off,f_sw=f_sw),)
        P_sw_I = ne.evaluate("where(P_sw_I_expr > 0.0, P_sw_I_expr, 0.0)", local_dict=dict(P_sw_I_expr=P_sw_I_expr))


        # Diode

        P_sw_D_expr = ne.evaluate("((c2 * (is_D * V_dc) / (I_ref * V_ref)) * Err_D * f_sw)",
                             local_dict=dict(c2=c2,V_dc=V_dc,is_D=is_D,f_sw=f_sw,I_ref=I_ref,V_ref=V_ref,Err_D=Err_D),)
        #P_sw_D = np.maximum(P_sw_D, 0)
        P_sw_D = ne.evaluate("where(P_sw_D_expr > 0.0, P_sw_D_expr, 0.0)", local_dict=dict(P_sw_D_expr=P_sw_D_expr))

        return P_sw_I, P_sw_D


    @staticmethod
    def Conduction_losses(is_I, R_IGBT, V_0_IGBT, M, pf, is_D, R_D, V_0_D):

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

        is_I = np.ascontiguousarray(is_I, dtype=np.float64)
        is_D = np.ascontiguousarray(is_D, dtype=np.float64)
        pf = np.ascontiguousarray(pf, dtype=np.float64)

        c1 = np.sqrt(2 * np.pi)
        c2 = (3 * np.pi)
        c3 = np.pi

        # IGBT

        #P_con_I = (((is_I ** 2 / 4.0) * R_IGBT) + ((is_I / np.sqrt(2 * np.pi)) * V_0_IGBT) +
        #           ((((is_I ** 2 / 4.0) * (8 * M / (3 * np.pi)) * R_IGBT) + (
        #                   (is_I / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_IGBT)) * abs(pf)))
        #P_con_I = np.maximum(P_con_I, 0)


        P_con_I_expr = ne.evaluate("(((is_I ** 2 / 4.0) * R_IGBT) + ((is_I / c1) * V_0_IGBT) + "
                              "((((is_I ** 2 / 4.0) * (8 * M / c2) * R_IGBT) +"
                              " ((is_I / c1) * (c3 * M / 4.0) * V_0_IGBT)) * abs(pf)))",
            local_dict=dict(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M, pf=pf, c1=c1, c2=c2, c3=c3),)
        P_con_I = ne.evaluate("where(P_con_I_expr > 0.0, P_con_I_expr, 0.0)", local_dict=dict(P_con_I_expr=P_con_I_expr))

        # Diode

        #P_con_D = ((((is_D ** 2 / 4.0) * R_D) + ((is_D / np.sqrt(2 * np.pi)) * V_0_D)) -
        #           ((((is_D ** 2 / 4.0)) * ((8 * M / (3 * np.pi)) * R_D)) + (
        #                       (is_D / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_D)) * abs(pf))
        #P_con_D = np.maximum(P_con_D, 0)

        # Diode

        P_con_D_expr = ne.evaluate("((((is_D ** 2 / 4.0) * R_D) + ((is_D / c1) * V_0_D)) -"
                              " ((((is_D ** 2 / 4.0)) * ((8 * M / c2) * R_D)) + "
                              "((is_D / c1) * (c3 * M / 4.0) * V_0_D)) * abs(pf))",
                                   local_dict=dict(is_D=is_D, R_D=R_D, V_0_D=V_0_D, M=M, pf=pf, c1=c1, c2=c2, c3=c3),)
        P_con_D = ne.evaluate("where(P_con_D_expr > 0.0, P_con_D_expr, 0.0)",local_dict=dict(P_con_D_expr=P_con_D_expr))

        return P_con_I, P_con_D


    @staticmethod
    def Cycles_to_failure(A,             # Input = float
                          alpha,         # Input = float
                          beta1,         # Input = float
                          beta0,         # Input = float
                          C,             # Input = float
                          gamma,         # Input = float
                          fd,            # Input = float
                          Ea,            # Input = float
                          k_b,           # Input = float
                          Tj_mean,       # Input = array
                          delta_Tj,      # Input = array
                          t_cycle_heat,  # Input = array
                          ar):           # Input = float

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
        

        Tj_mean   = np.ascontiguousarray(Tj_mean,   dtype=np.float64)
        delta_Tj  = np.ascontiguousarray(delta_Tj,  dtype=np.float64)
        t_cycle_heat = np.ascontiguousarray(t_cycle_heat, dtype=np.float64)

        if np.any(Tj_mean <= 0):
            raise ValueError(
                "Tj_mean contains 0 K or negative values, which is not physically possible. Check input data.")

        c1 = (np.exp(Ea / (k_b * Tj_mean)))

        Nf = ne.evaluate("(A) * "
                         "((delta_Tj) ** alpha) * "
                         "((ar) ** ((beta1 * delta_Tj) + beta0)) * "
                         "((C + ((t_cycle_heat) ** gamma)) / (C + 1)) *"
                         "c1 * "
                         "(fd)", local_dict=dict(A=A,
                                                 alpha=alpha,
                                                 beta1=beta1,
                                                 beta0=beta0,
                                                 C=C,
                                                 gamma=gamma,
                                                 fd=fd,
                                                 delta_Tj=delta_Tj,
                                                 ar=ar,
                                                 t_cycle_heat=t_cycle_heat,
                                                 c1=c1) )
        return Nf


    @staticmethod
    def window_stats(temp, time_window, steps_per_sec, pf):

        """
         Compute window-based thermal statistics.

         Parameters
         ----------
         temp : ndarray
             Junction temperature time series (K).
         time_window : int
             Window length in seconds over which stats are computed.
         steps_per_sec : int
             Number of simulation steps per second (time resolution).
         pf : ndarray
             Power factor array, used to align time indices.

         Returns
         -------
         mean_T : ndarray
             Mean junction temperature in each window (K).
         delta_T : ndarray
             Temperature swing (Tmax - Tmin) per window (K).
         t_cycle_heat : ndarray
             Duration of each heating cycle within a window (s).
         time_period_df2 : ndarray
             Array of window time indices aligned with pf series.
         """

        n = len(temp)
        window_size = int(round(time_window * steps_per_sec))
        n_windows = n // window_size

        data = temp[:n_windows * window_size].reshape(n_windows, window_size)

        mean_T = data.mean(axis=1)

        Tmax = data.max(axis=1)
        Tmin = data.min(axis=1)
        delta_T = Tmax - Tmin

        idx_max = data.argmax(axis=1)
        idx_min = data.argmin(axis=1)
        t_cycle_heat = np.abs(idx_max - idx_min) / steps_per_sec

        time_period_df2 = np.arange(0, len(pf), time_window)

        return mean_T, delta_T, t_cycle_heat, time_period_df2

    @staticmethod
    def check_vce(overshoot_margin, V_dc, max_V_CE):
        """
        Check if collector–emitter voltage exceeds the maximum allowed value.

        Parameters
        ----------
        overshoot_margin : float
            Overshoot margin (fraction or percentage in decimal form).
        V_dc : float or np.ndarray
            DC link voltage (single value or array of values).
        max_V_CE : float
            Maximum allowed collector–emitter voltage.

        Raises
        ------
        UserWarning
            If the collector–emitter voltage exceeds max_V_CE.
        """
        V_CE = V_dc * (1 + overshoot_margin)

        violations = np.where(V_CE > max_V_CE)[0]

        if violations.size > 0:
            raise ValueError(
                f"Collector–emitter voltage exceeded! "
                f"Maximum allowed: {max_V_CE}, but got values up to {V_CE.max()}. "
                "Please reduce the DC voltage or update the IGBT specifications."
            )

        return V_CE

    @staticmethod

    def check_igbt_diode_limits(
            is_I, is_D, T_j_I, T_j_D,
            max_IGBT_RMS_Current, max_IGBT_peak_Current,
            max_Diode_RMS_Current, max_Diode_peak_Current,
            max_IGBT_temperature, max_Diode_temperature
    ):
        """
        Check IGBT and Diode current and temperature limits.

        Parameters
        ----------
        is_I : np.ndarray
            IGBT current waveform.
        is_D : np.ndarray
            Diode current waveform.
        T_j_I : np.ndarray
            IGBT junction temperature (K).
        T_j_D : np.ndarray
            Diode junction temperature (K).
        max_IGBT_RMS_Current : float
            Maximum allowed IGBT RMS current (A).
        max_IGBT_peak_Current : float
            Maximum allowed IGBT peak current (A).
        max_Diode_RMS_Current : float
            Maximum allowed Diode RMS current (A).
        max_Diode_peak_Current : float
            Maximum allowed Diode peak current (A).
        max_IGBT_temperature : float
            Maximum allowed IGBT junction temperature (K).
        max_Diode_temperature : float
            Maximum allowed Diode junction temperature (K).
        sec_idx : int, optional
            Index or time step indicator for diagnostics (default=0).

        Raises
        ------
        Raises Error
            If any violation occurs.
        """

        # --- Compute RMS and Peak currents ---
        I_rms_is_I = np.sqrt(np.mean(is_I ** 2))
        I_peak_is_I = np.max(np.abs(is_I))
        I_rms_is_D = np.sqrt(np.mean(is_D ** 2))
        I_peak_is_D = np.max(np.abs(is_D))



        # --- IGBT Checks ---
        if I_rms_is_I > max_IGBT_RMS_Current:

            raise ValueError(
                f"IGBT RMS current exceeded! "
                f"Maximum allowed: {max_IGBT_RMS_Current:.2f} A, "
                f"but got {I_rms_is_I:.2f} A. "
                "Please reduce the power requirements or update the IGBT specifications."
            )

        if I_peak_is_I > max_IGBT_peak_Current:
            raise ValueError(
                f"IGBT peak current exceeded! "
                f"Maximum allowed: {max_IGBT_peak_Current:.2f} A, "
                f"but got {I_peak_is_I:.2f} A. "
                "Please reduce the power requirements or update the IGBT specifications."
            )

        # --- Diode Checks ---
        if I_rms_is_D > max_Diode_RMS_Current:
            raise ValueError(
                f"Diode RMS current exceeded! "
                f"Maximum allowed: {max_Diode_RMS_Current:.2f} A, "
                f"but got {I_rms_is_D:.2f} A. "
                "Please reduce the power requirements or update the diode specifications."
            )

        if I_peak_is_D > max_Diode_peak_Current:
            raise ValueError(
                f"Diode peak current exceeded! "
                f"Maximum allowed: {max_Diode_peak_Current:.2f} A, "
                f"but got {I_peak_is_D:.2f} A. "
                "Please reduce the power requirements or update the diode specifications."
            )

        # --- Temperature Checks ---
        if np.any(T_j_I > max_IGBT_temperature):
            raise ValueError(
                f"IGBT junction temperature exceeded! "
                f"Maximum allowed: {max_IGBT_temperature:.2f} K, "
                f"but got up to {np.max(T_j_I):.2f} K. "
                "Please improve heat dissipation (cooling), reduce losses, "
                "or update the IGBT specifications."
            )

        if np.any(T_j_D > max_Diode_temperature):
            raise ValueError(
                f"Diode junction temperature exceeded! "
                f"Maximum allowed: {max_Diode_temperature:.2f} K, "
                f"but got up to {np.max(T_j_D):.2f} K. "
                "Please improve heat dissipation (cooling), reduce losses, "
                "or update the diode specifications."
            )

    @staticmethod
    def Lifecycle_calculation(Number_of_cycles,pf):
        """
        Calculate device lifetime using Miner’s rule.

        Parameters
        ----------
        Number_of_cycles : array-like
            List/array of cycles-to-failure values (Nf) for each
            stress cycle severity. These are the Nf values you
            already computed for the device (IGBT or diode).
        pf : array-like
            Power factor values per sec.

        Returns
        -------
        Life_period : float
            Estimated life of the device in years.

        Notes
        -----
        - Miner’s rule: cumulative damage D = sum(1/Nf_i).
        - Damage per pass = sum(1/Nf_i).
        - Repetitions to failure = 1 / Damage_per_set.
        - Total cycles to failure = len(Nf) * Repetitions.
        - Life_period = Total_cycles / (seconds in 50 years).
        """

        Number_of_cycles = np.ascontiguousarray(Number_of_cycles, dtype=np.float64)

        Damage_per_set = np.sum(np.reciprocal(Number_of_cycles, dtype=np.float64))

        #Total_cycles = len(Number_of_cycles) * (1 / Damage_per_set)
        #Life_period = Total_cycles / (3600 * 24 * 365 * (len(Number_of_cycles) / len(pf)))

        return len(pf)/(3600*24*365*Damage_per_set)

    @staticmethod
    def delta_t_calculations(A,             # Input = float
                             alpha,         # Input = float
                             beta1,         # Input = float
                             beta0,         # Input = float
                             C,             # Input = float
                             gamma,         # Input = float
                             fd,            # Input = float
                             Ea,            # Input = float
                             k_b,           # Input = float
                             Tj_mean,       # Input = array
                             t_cycle_float, # Input = float
                             ar,            # Input = float
                             Nf,            # Input = array
                             pf,            # Input = array
                             Life):         # Input = float

        """

        Compute the equivalent delta_Tj (temperature swing) for reliability assessment,
        following the approach in the lifetime evaluation paper.

        Parameters
        ----------
        A : float
            Lifetime model coefficient (prefactor).
        alpha : float
            Temperature exponent in the lifetime model.
        beta1 : float
            Coefficient for ΔTj dependence in the lifetime model.
        beta0 : float
            Offset term in the ΔTj exponent of the lifetime model.
        C : float
            Constant used in the cycle-duration correction factor.
        gamma : float
            Exponent for the cycle-duration correction.
        fd : float
            Device-specific correction factor (empirical).
        Ea : float
            Activation energy [eV].
        k_b : float
            Boltzmann constant [eV/K].
        Tj_mean : array
            Array of mean junction temperatures [K].
        t_cycle_float : float
            Equivalent cycle duration [s].
        ar : float
            Reference amplitude ratio (empirical model parameter).
        Nf : array
            Cycles-to-failure values from the time-series analysis.
        pf : array
            Power factor time-series (used to normalize cycle counts).
        Life : float
            Expected device lifetime (years).

        Returns
        -------
        number_of_yearly_cycles : float
            Total number of equivalent thermal cycles per year.
        Yearly_life_consumption_I : float
            Annual fraction of life consumed (1/Life).
        Tj_mean_float : float
            Mean junction temperature (averaged over all cycles).
        delta_Tj_float : float
            Equivalent static temperature swing ΔTj' [K] computed from Lambert W formulation.
        t_cycle_float : float
            Equivalent cycle duration [s] .

        """

        Tj_mean = np.ascontiguousarray(Tj_mean, dtype=np.float64)
        Tj_mean_float = Tj_mean.mean()

        number_of_yearly_cycles = float(3600 * 24 * 365 * (len(Nf) / len(pf)))
        Yearly_life_consumption_I = (1 / Life)

        #delta_Tj_float = ((alpha / (beta1 * np.log(ar))) * (lambertw((beta1 * np.log(ar) / alpha) * ((((( number_of_yearly_cycles / Yearly_life_consumption_I) / (A * ((C + ((t_cycle_float) ** gamma)) / (C + 1)) * (np.exp(Ea / (k_b * Tj_mean_float))) * fd)) * ar ** (-beta0))) ** (1 / alpha)))))

        ln_ar = np.log(ar)
        exp_term = (np.exp(Ea / (k_b * Tj_mean_float)))
        lambertw_term = ((beta1 * ln_ar / alpha) * (((((number_of_yearly_cycles / Yearly_life_consumption_I) / (A * ((C +((t_cycle_float) ** gamma)) / (C + 1)) * exp_term * fd)) * ar ** (-beta0))) ** (1 / alpha)))
        delta_Tj_float = (alpha / (beta1 * ln_ar)) * lambertw(lambertw_term, k=0).real


        return number_of_yearly_cycles, Yearly_life_consumption_I, Tj_mean_float, delta_Tj_float, t_cycle_float

    @staticmethod
    def variable_input_normal_distribution(variable, normal_distribution, number_of_samples):

        """
            Generate random samples of a variable using a normal distribution
            centered on its nominal value, with variability expressed as a
            fraction of the variable (e.g. ±5%).

            Parameters
            ----------
            variable : float
                Nominal (mean) value of the parameter to be randomized.
            normal_distribution : float
                Relative standard deviation as a fraction of the nominal value
                (e.g., 0.05 for ±5% variation).
            number_of_samples : int
                Number of random samples to generate.

            Returns
            -------
            samples : np.ndarray
                Array of normally distributed random samples centered at `variable`
                with standard deviation = `normal_distribution * abs(variable)`.
            """

        sigma = normal_distribution * abs(variable)
        samples = np.random.normal(variable, sigma, number_of_samples)
        return samples

    @staticmethod
    def cdf_with_B_lines(ax, samples, label, title):

        """
        Plot the empirical cumulative distribution function (CDF) of Monte Carlo
        lifetime samples and annotate B1 and B10 reliability points with red
        dotted lines and labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis object on which the CDF will be plotted.
        samples : np.ndarray
            Array of lifetime samples (e.g., years to failure) from Monte Carlo.
        label : str
            Label for the plotted CDF line (e.g., "IGBT", "Diode").
        title : str
            Title string for the figure.

        Returns
        -------
        None
            The function modifies the given axis by plotting the CDF curve,
            drawing B1/B10 markers, and adding annotations.
        """

        x = np.sort(samples)
        y = np.arange(1, len(x) + 1) / len(x) * 100.0
        ax.plot(x, y, label=label)

        # Ensure 0–100% y-range for clarity
        ax.set_ylim(0, 100)

        # Compute B1 and B10 (1st and 10th percentiles)
        b_marks = {1: np.percentile(samples, 1),
                   10: np.percentile(samples, 10),
                   50: np.percentile(samples, 50)}

        # Draw red dotted lines + intersection markers + annotations
        for bx, xv in b_marks.items():
            ax.axvline(xv, color='red', linestyle=':', linewidth=1)
            ax.axhline(bx, color='red', linestyle=':', linewidth=1)
            ax.plot([xv], [bx], 'o', color='red', markersize=4)  # intersection dot
            ax.annotate(f"B{bx} = {xv:.2f} yrs",
                        xy=(xv, bx),
                        xytext=(6, 6),
                        textcoords='offset points',
                        fontsize=10,
                        color='red',
                        fontweight='bold')

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Cumulative failure probability (%)")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="best")

    @staticmethod
    def check_max_apparent_power_switch(S,Vs, max_IGBT_RMS_Current, inverter_phases):

        """
        Check that inverter apparent power does not exceed device current limits.

        Parameters
        ----------
        S : array
            Apparent power demand per sample [VA].
        Vs : array
            RMS AC-side phase voltage [V].
        max_IGBT_RMS_Current : float
            Maximum allowable IGBT RMS current per phase [A].
        inverter_phases : int
            Number of inverter phases (1 for single-phase, 3 for three-phase).

        Raises
        ------
        ValueError
            If any sample of S exceeds the maximum apparent power capability
            (S_max = Vs * max_IGBT_RMS_Current * inverter_phases).
        """

        S_max = Vs * max_IGBT_RMS_Current * inverter_phases
        if np.any(S > S_max):
            raise ValueError(
                f"Apparent power limit exceeded: "
                f"S_max = {np.max(Vs) * max_IGBT_RMS_Current * inverter_phases} VA. "
                f"Reduce the apparent power requirement from the inverter.")

    @staticmethod
    def Lifecycle_calculation_acceleration_factor(Nf ,pf , Component_max_lifetime ):

        """
        Compute a finite, mission-referenced lifetime from per-cycle physics (Nf) using
        a Table-of-Damage style normalization.

        -----------------------------
        WHAT THIS FUNCTION RETURNS
        -----------------------------
        Life_I : float
            The *equivalent* lifetime (in years) of the component when operated under the
            provided profile window `pf`, *normalized* to the mission life `Component_max_lifetime`.
            - If the present operating profile is harsher than the mission, Life_I < Component_max_lifetime.
            - If it is milder, Life_I > Component_max_lifetime (unless the calendar floor dominates).

        -----------------------------
        KEY IDEAS / WHY THIS WORKS
        -----------------------------

        1) Miner’s rule in “per-set” form:
            We consider a single execution of your profile window which is one sinusoidal cycle to be one “set”.
            The per-set damage is sum(1/Nf). Repeating that set many times accumulates damage.

        2) Time normalization (sets per year):
            We convert one set to *years* by computing how many times per year the set repeats.
            This ties the Miner sum to calendar time (no more infinite lifetimes).

        3) Calendar-damage floor equals mission life:
            We add a tiny baseline (calendar) damage per set so life never diverges at very low stress.
            Here the floor is chosen to be exactly the mission (`Component_max_lifetime`), meaning
            “no-better-than-mission” behavior at the extreme low-stress limit.

        4) Acceleration Factor (AF) vs. mission:
            We compare the *effective* per-set damage to the *target* per-set damage implied by the
            mission life. AF = damage_test / damage_target. Life = mission / AF.
            This is exactly the “Table of Damage” normalization.

        -----------------------------
        PARAMETERS (INPUTS)
        -----------------------------
        Nf : array-like of float
            Cycles-to-failure for each counted damage event inside one pf window (one “set”).

        pf : sequence (e.g., array, list)
            The time-series profile window. Its length is used as a proxy for
            “seconds per set” under the assumption of 1 Hz sampling (1 sample = 1 second).

        Component_max_lifetime : float
            The mission/reference life in years (e.g., 30 years). Used in two roles:
            (a) the *target* lifetime for Table-of-Damage normalization, and
            (b) the *calendar floor* (no-better-than-mission) so life cannot diverge at very low stress.

        -----------------------------
        OUTPUT
        -----------------------------
        Life_I : float
            Mission-referenced equivalent lifetime in years.
        """

        Nf = np.asarray(Nf, dtype=float)

        # --- Miner damage per set (include the events inside each pf step) ---
        # If events_per_step varies with time, make it an array shaped like Nf.
        damage_per_set = float(np.sum(1 / Nf))  # sum_i (events_in_step_i / Nf)
        # --- correct time scaling to sets/year ---
        sets_per_year = (3600.0 * 24.0 * 365.0) / (float(len(pf)))
        floor_damage_per_set = 1.0 / (sets_per_year * Component_max_lifetime)
        effective_damage_per_set = damage_per_set + floor_damage_per_set  # SO this is total damage
        target_damage_per_set = (1.0 / Component_max_lifetime) / sets_per_year
        AF = effective_damage_per_set / target_damage_per_set
        Life = Component_max_lifetime / AF

        return Life

    @staticmethod
    def Lifecycle_normal_distribution_calculation_acceleration_factor(Nf ,f ,Component_max_lifetime ):

        """
        Parameters
        ----------
        Nf : float
            Number of cycles to failure for the component under given conditions [-].
        f : float
            Operating frequency of the component [Hz].
        Component_max_lifetime : float
            Maximum component lifetime in years.

        Returns
        -------
        Life_period : float
            Effective lifetime of the component in years, after applying
            the acceleration factor.
        """

        damage_per_set = 1 / Nf
        sets_per_year = 3600.0 * 24.0 * 365.0 * f
        floor_damage_per_set = 1.0 / (sets_per_year * Component_max_lifetime)
        effective_damage_per_set = damage_per_set + floor_damage_per_set
        target_damage_per_set = (1.0 / Component_max_lifetime) / sets_per_year
        AF = effective_damage_per_set / target_damage_per_set
        Life_period = Component_max_lifetime / AF

        return Life_period

    @staticmethod
    def free_ram_now():

        """
        Function to explicitly free RAM
        Calls Python's garbage collector and, on Linux systems, forces glibc to
        return freed memory pages to the operating system.
        """

        gc.collect()
        try:
            if sys.platform.startswith("linux"):
                ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    @staticmethod
    def cat(lst):
        """
        # Function to concatenate list of numpy arrays into a single numpy array
        """
        return np.concatenate(lst)

    @staticmethod
    def find_sorted_files(base_dir: str, prefix: str, recursive: bool = False):
        """
        Find parquet files named '<prefix>_*.parquet' under base_dir.
        - If recursive=True, search subdirectories too.
        - Returns files sorted by numeric chunk suffix.
        """

        def _extract_chunk_no(path: str, prefix: str) -> int:
            """
            Extract trailing chunk number from '<prefix>_<N>.parquet' (e.g., 'df_1_12.parquet').
            Returns a large number for non-matching names so they sort to the end.
            """
            m = re.search(rf"{re.escape(prefix)}_(\d+)\.parquet$", os.path.basename(path))
            return int(m.group(1)) if m else 10 ** 9

        pattern = os.path.join(base_dir, "**" if recursive else "", f"{prefix}_*.parquet")
        files = glob.glob(pattern, recursive=recursive)
        files = [f for f in files if os.path.isfile(f)]
        files.sort(key=lambda p: _extract_chunk_no(p, prefix))
        return files

    @staticmethod
    def merge_parquet_files(files, out_path: str, compression: str = "zstd", validate_schema: bool = True):
        """
        Stream-merge multiple parquet files into a single parquet at out_path.

        - validate_schema=True: require identical schemas; raises if mismatch.
          If False, will write using the first file's schema and attempt to select
          those columns from subsequent files (columns missing will raise).
        """
        if not files:
            # print(f"[merge] No input files found for {out_path}. Skipping.")
            return

        # Schema from the first file
        first_pf = pq.ParquetFile(files[0])
        base_schema = first_pf.schema_arrow

        if validate_schema:
            # Ensure all schemas match exactly
            for fp in files[1:]:
                sch = pq.ParquetFile(fp).schema_arrow
                if sch != base_schema:
                    raise ValueError(f"[merge] Schema mismatch between '{files[0]}' and '{fp}'. "
                                     f"Set validate_schema=False to attempt a permissive merge.")
            writer = pq.ParquetWriter(out_path, base_schema, compression=compression)
            total_rows = 0
            try:
                for fp in files:
                    pf = pq.ParquetFile(fp)
                    for batch in pf.iter_batches():
                        writer.write_batch(batch)
                        total_rows += batch.num_rows
            finally:
                writer.close()
            # print(f"[merge] Wrote {total_rows} rows to: {out_path}")
            return

        # Permissive path: use columns from the first file's schema
        base_cols = [f.name for f in base_schema]
        writer = pq.ParquetWriter(out_path, base_schema, compression=compression)
        total_rows = 0
        try:
            for fp in files:
                pf = pq.ParquetFile(fp)
                # Build a Table with the base columns only (order preserved)
                # tbl = pf.read(columns=base_cols)
                # Stream out in record batches so we keep memory flat
                # for batch in tbl.to_batches():
                #    writer.write_batch(batch)
                #    total_rows += batch.num_rows
                for batch in pf.iter_batches(columns=base_cols):
                    writer.write_batch(batch)
        finally:
            writer.close()

    @staticmethod
    def latest_chunk_file(base_dir: str, prefix: str, recursive: bool = False):
        """
        Return the full path to the latest (highest chunk number) file for a prefix,
        or None if none found.
        """

        def find_sorted_files(base_dir: str, prefix: str, recursive: bool = False):
            """
            Find parquet files named '<prefix>_*.parquet' under base_dir.
            - If recursive=True, search subdirectories too.
            - Returns files sorted by numeric chunk suffix.
            """

            def _extract_chunk_no(path: str, prefix: str) -> int:
                """
                Extract trailing chunk number from '<prefix>_<N>.parquet' (e.g., 'df_1_12.parquet').
                Returns a large number for non-matching names so they sort to the end.
                """
                m = re.search(rf"{re.escape(prefix)}_(\d+)\.parquet$", os.path.basename(path))
                return int(m.group(1)) if m else 10 ** 9

            pattern = os.path.join(base_dir, "**" if recursive else "", f"{prefix}_*.parquet")
            files = glob.glob(pattern, recursive=recursive)
            files = [f for f in files if os.path.isfile(f)]
            files.sort(key=lambda p: _extract_chunk_no(p, prefix))
            return files

        files = find_sorted_files(base_dir, prefix, recursive=recursive)
        return files[-1] if files else None

    @staticmethod
    def load_latest_df(prefix, base_dir):
        """
        Find and load the parquet with the highest chunk number for a given prefix (df_1, df_2, …).
        Expects chunk files in subfolder: base_dir/prefix/
        """
        chunk_dir = os.path.join(base_dir, prefix)  # e.g., Location_dataframes/df_1
        files = glob.glob(os.path.join(chunk_dir, f"{prefix}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {prefix}_*.parquet files found in {chunk_dir}")

        def extract_chunk_no(path):
            m = re.search(rf"{prefix}_(\d+)\.parquet$", os.path.basename(path))
            return int(m.group(1)) if m else -1

        latest_file = max(files, key=extract_chunk_no)
        # print(f"[INFO] Loading latest {prefix} file: {latest_file}")
        return pd.read_parquet(latest_file, engine="pyarrow")


    @staticmethod
    def resize_foster_branches(r, tau, N):
        """
        Resample Foster RC branches to fixed length N using log-time interpolation.
        Preserves total Rθ and approximates the transient shape.
        """
        import numpy as np

        r = np.asarray(r, dtype=np.float64).ravel()
        tau = np.asarray(tau, dtype=np.float64).ravel()
        assert r.size == tau.size and r.ndim == 1

        m = r.size
        if m == N:
            return r.copy(), tau.copy()

        if m == 1:
            # Expand one lump to N via geometric spread around tau0; split Rθ equally.
            r0, tau0 = r[0], tau[0]
            scales = np.geomspace(0.2, 5.0, N)  # tweak spread if you want
            tau_new = tau0 * scales
            r_new = np.full(N, r0 / N, dtype=np.float64)
            return r_new, tau_new

        # m >= 2: interpolate cumulative Rθ over log(τ)
        idx = np.argsort(tau)
        r_s = r[idx]
        tau_s = tau[idx]
        logtau = np.log(tau_s)
        Rcum = np.cumsum(r_s)

        logtau_new = np.linspace(logtau[0], logtau[-1], N)
        Rcum_new = np.interp(logtau_new, logtau, Rcum)

        r_new = np.empty(N, dtype=np.float64)
        r_new[0] = Rcum_new[0]
        r_new[1:] = Rcum_new[1:] - Rcum_new[:-1]
        r_new[r_new < 0.0] = 0.0  # guard tiny negatives
        # renormalize exactly
        s = r_new.sum()
        if s > 0:
            r_new *= (Rcum[-1] / s)
        tau_new = np.exp(logtau_new)
        return r_new, tau_new
