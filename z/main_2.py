import numpy as np
import pandas as pd
from Input_parameters_file import Input_parameters_class
from Calculation_functions_file import Calculation_functions_class
from Electro_thermal_behavior_file import Electro_thermal_behavior_class
import time
from datetime import datetime
from Dataframe_saving_file import save_dataframes
from Plotting_file import Plotting_class
import os

Electro_thermal_behavior_class = Electro_thermal_behavior_class()
Calculation_functions_class = Calculation_functions_class()

def main_2(P=None,pf=None,Q=None,Loadprofile_name=None):

    Input_parameters = Input_parameters_class(P=P, pf=pf,Q=Q)

    start_time = time.time()

    '################################################################################################################################################################'
    'Input Parameters'
    '################################################################################################################################################################'

    # ----------------------------------------#
    # Power flow values
    # ----------------------------------------#

    pf = Input_parameters.pf
    P = Input_parameters.P
    Q = Input_parameters.Q
    V_dc = Input_parameters.V_dc
    Vs = Input_parameters.Vs
    f = Input_parameters.f
    M = Input_parameters.M
    Tamb = Input_parameters.Tamb
    dt = Input_parameters.dt

    # ----------------------------------------#
    # Inverter Design options
    # ----------------------------------------#

    thermal_states = Input_parameters.thermal_states
    single_phase_inverter_topology = Input_parameters.single_phase_inverter_topology
    inverter_phases = Input_parameters.inverter_phases
    modulation_scheme = Input_parameters.modulation_scheme
    overshoot_margin_inverter = Input_parameters.overshoot_margin_inverter
    design_control = Input_parameters.design_control

    # ----------------------------------------#
    # Max switch limits
    # ----------------------------------------#

    max_V_CE = Input_parameters.max_V_CE
    max_IGBT_RMS_Current = Input_parameters.max_IGBT_RMS_Current
    max_IGBT_peak_Current = Input_parameters.max_IGBT_peak_Current
    max_Diode_RMS_Current = Input_parameters.max_Diode_RMS_Current
    max_Diode_peak_Current = Input_parameters.max_Diode_peak_Current
    max_IGBT_temperature = Input_parameters.max_IGBT_temperature
    max_Diode_temperature = Input_parameters.max_Diode_temperature
    overshoot_margin = Input_parameters.overshoot_margin

    # ----------------------------------------#
    # Max lifetime
    # ----------------------------------------#

    IGBT_max_lifetime  = Input_parameters.IGBT_max_lifetime
    Diode_max_lifetime  = Input_parameters.Diode_max_lifetime

    #----------------------------------------#
    # Switching losses
    #----------------------------------------#

    # IGBT
    f_sw = Input_parameters.f_sw
    t_on = Input_parameters.t_on
    t_off = Input_parameters.t_off

    # Diode
    I_ref = Input_parameters.I_ref
    V_ref = Input_parameters.V_ref
    Err_D = Input_parameters.Err_D

    #----------------------------------------#
    # Conduction losses
    #----------------------------------------#

    # IGBT
    R_IGBT  = Input_parameters.R_IGBT
    V_0_IGBT = Input_parameters.V_0_IGBT

    # Diode
    V_0_D = Input_parameters.V_0_D
    R_D = Input_parameters.R_D

    #----------------------------------------#
    # Thermal Parameters
    #----------------------------------------#

    # Paste
    r_paste = Input_parameters.r_paste
    tau_paste = Input_parameters.tau_paste

    # Heat Sink
    r_sink = Input_parameters.r_sink
    tau_sink = Input_parameters.tau_sink

    # IGBT
    r_I = Input_parameters.r_I
    tau_I = Input_parameters.tau_I

    # Diode
    r_D = Input_parameters.r_D
    tau_D = Input_parameters.tau_D

    # ----------------------------------------#
    # Input_parameters for calculating cycles
    # ----------------------------------------#

    A = Input_parameters.A
    alpha = Input_parameters.alpha
    beta1 = Input_parameters.beta1
    beta0 = Input_parameters.beta0
    C = Input_parameters.C
    gamma = Input_parameters.gamma
    fd = Input_parameters.fd
    fI = Input_parameters.fI
    Ea = Input_parameters.Ea
    k_b = Input_parameters.k_b
    ar = Input_parameters.ar

    # ----------------------------------------#
    # Thermal state & constants
    # ----------------------------------------#

    alpha_I = Input_parameters.alpha_I
    alpha_D = Input_parameters.alpha_D
    alpha_p = Input_parameters.alpha_p
    alpha_s = Input_parameters.alpha_s

    #----------------------------------------#
    # Others
    #----------------------------------------#

    omega = Input_parameters.omega
    Time_period = Input_parameters.Time_period
    Tgrid = Input_parameters.Tgrid
    Nsec = Input_parameters.Nsec
    Ngrid = Input_parameters.Ngrid

    # ----------------------------------------#
    # Code related options
    # ----------------------------------------#

    saving_dataframes = Input_parameters.saving_dataframes
    plotting_values = Input_parameters.plotting_values
    Location_dataframes = Input_parameters.Location_dataframes
    chunk_seconds = Input_parameters.chunk_seconds

    # ----------------------------------------#
    # Miscellaneous
    # ----------------------------------------#

    t_cycle_heat_my_value = Input_parameters.t_cycle_heat_my_value

    '################################################################################################################################################################'
    'main'
    '################################################################################################################################################################'

    #----------------------------------------#
    # Checking collector–emitter voltage
    #----------------------------------------#

    Calculation_functions_class.check_vce(overshoot_margin, V_dc, max_V_CE)

    #----------------------------------------#
    # Calculate power flow equations
    #----------------------------------------#

    S,Is,phi,P,Q,Vs = Calculation_functions_class.compute_power_flow_from_pf(P=P,
                                                                             Q=Q,
                                                                             V_dc = V_dc,
                                                                             pf=pf,
                                                                             Vs=Vs,
                                                                             M=M,
                                                                             single_phase_inverter_topology=single_phase_inverter_topology,
                                                                             inverter_phases= inverter_phases,
                                                                             modulation_scheme=modulation_scheme)  # Inverter Power Flow

    if design_control == "inverter":
        S,Is,phi,P,Q,Vs,N_parallel = Calculation_functions_class.compute_power_flow_from_pf_design_control_inverter(overshoot_margin_inverter,
                                                           inverter_phases=inverter_phases,
                                                           Vs=Vs,
                                                           max_IGBT_RMS_Current=max_IGBT_RMS_Current,
                                                           S=S,
                                                           P=P,
                                                           Q=Q,
                                                           pf=pf,
                                                           single_phase_inverter_topology=single_phase_inverter_topology,
                                                           modulation_scheme=modulation_scheme,
                                                           M=M,
                                                           V_dc=V_dc)
    elif design_control == "switch":
        N_parallel = 1

    Calculation_functions_class.check_max_apparent_power_switch(S=S,Vs=Vs,max_IGBT_RMS_Current=max_IGBT_RMS_Current,inverter_phases=inverter_phases)

    #----------------------------------------#
    # Impedance calculations
    #----------------------------------------#

    # Make constants contiguous float64 ONCE
    alpha_I = np.ascontiguousarray(alpha_I, dtype=np.float64)
    alpha_D = np.ascontiguousarray(alpha_D, dtype=np.float64)
    alpha_p = np.ascontiguousarray(alpha_p, dtype=np.float64)
    alpha_s = np.ascontiguousarray(alpha_s, dtype=np.float64)

    r_I     = np.ascontiguousarray(r_I,     dtype=np.float64)
    r_D     = np.ascontiguousarray(r_D,     dtype=np.float64)
    r_paste = np.ascontiguousarray(r_paste, dtype=np.float64)
    r_sink  = np.ascontiguousarray(r_sink,  dtype=np.float64)

    #----------------------------------------#
    # Temperature calculations
    #----------------------------------------#

    Tbr_I = np.zeros_like(r_I, dtype=float)     # [K] Temperature rise contributions of each Foster RC branch for IGBT junction → case.
    Tbr_D = np.zeros_like(r_D, dtype=float)     # [K] Temperature rise contributions of each Foster RC branch for diode junction → case.

    # Shared State
    Tbr_p = np.zeros_like(r_paste, dtype=float) # [K] Temperature rise across the thermal paste (case → heatsink interface).
    Tbr_s = np.zeros_like(r_sink, dtype=float)  # [K] Temperature rise contributions of each Foster RC branch for the heatsink (sink → ambient).

    # Separated_state
    Tbr_p_I = np.zeros_like(r_paste, dtype=float) # [K] Temperature rise across the thermal paste (case → heatsink interface) in IGBT.
    Tbr_s_I = np.zeros_like(r_sink, dtype=float)  # [K] Temperature rise contributions of each Foster RC branch for the heatsink (sink → ambient) in IGBT.
    Tbr_p_D = np.zeros_like(r_paste, dtype=float) # [K] Temperature rise across the thermal paste (case → heatsink interface) in Diode.
    Tbr_s_D = np.zeros_like(r_sink, dtype=float)  # [K] Temperature rise contributions of each Foster RC branch for the heatsink (sink → ambient) in Diode.

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Code_name = "main_2"
    if Loadprofile_name==None:
        Loadprofile_name = "Loadprofile"
    timestamp = f"{Code_name}_{Loadprofile_name}_{timestamp}"
    num_secs = len(pf)

    Location_dataframes = f"{Location_dataframes}/{timestamp}"
    os.makedirs(Location_dataframes, exist_ok=True)

    df1_dir = os.path.join(Location_dataframes, "df_1")
    df2_dir = os.path.join(Location_dataframes, "df_2")
    os.makedirs(df1_dir, exist_ok=True)
    os.makedirs(df2_dir, exist_ok=True)

    print("Total chunks :",int(num_secs/chunk_seconds))

    for chunk_start in range(0, num_secs, chunk_seconds):
        chunk_end = min(chunk_start + chunk_seconds, num_secs)
        chunk_no = (chunk_start // chunk_seconds) + 1  # 1-based chunk index
        is_last_chunk = (chunk_end == num_secs)

        start_time_chunk = time.time()

        print("Chunk Number : ",chunk_no)

        # slice inputs for the chunk (pf also sliced for df_2_2 windowing)
        Vs_chunk = Vs[chunk_start:chunk_end]
        Is_chunk = Is[chunk_start:chunk_end]
        phi_chunk = phi[chunk_start:chunk_end]
        Vdc_chunk = V_dc[chunk_start:chunk_end]
        pf_chunk = pf[chunk_start:chunk_end]

        TjI_list = []
        TjD_list = []

        if is_last_chunk:
            sec_idx_list = []
            time_list = []
            m_list = []
            is_I_list = []
            is_D_list = []
            P_sw_I_list = []
            P_sw_D_list = []
            P_con_I_list = []
            P_con_D_list = []
            P_leg_list = []
            vs_list = []
            is_inv_list = []

        if thermal_states == "separated":

            for sec_idx, (Vs_i, Is_i, phi_i, Vdc_i, pf_i) in enumerate(zip(Vs_chunk, Is_chunk, phi_chunk, Vdc_chunk, pf_chunk),start=chunk_start):
                t, m, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D, P_leg, T_j_I, T_j_D, vs_inverter, is_inverter, Tbr_I, Tbr_D, Tbr_p_I, Tbr_s_I, Tbr_p_D, Tbr_s_D = (
                    Electro_thermal_behavior_class.Electro_thermal_behavior_separated_thermal_state(dt=dt,             # Input = Float
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
                                                                                                    alpha_s=alpha_s,   # Input = Array
                                                                                                    r_I=r_I,           # Input = Array
                                                                                                    r_D=r_D,           # Input = Array
                                                                                                    r_paste=r_paste,   # Input = Array
                                                                                                    r_sink=r_sink,     # Input = Array
                                                                                                    Tamb=Tamb,         # Input = Array
                                                                                                    Tbr_I=Tbr_I,       # Input = Array
                                                                                                    Tbr_D=Tbr_D,       # Input = Array
                                                                                                    Tbr_p_I=Tbr_p_I,   # Input = Array
                                                                                                    Tbr_s_I=Tbr_s_I,   # Input = Array
                                                                                                    Tbr_p_D=Tbr_p_D,   # Input = Array
                                                                                                    Tbr_s_D=Tbr_s_D))  # Input = Array

                Calculation_functions_class.check_igbt_diode_limits(
                    is_I=is_I, is_D=is_D, T_j_I=T_j_I, T_j_D=T_j_D,
                    max_IGBT_RMS_Current=max_IGBT_RMS_Current,
                    max_IGBT_peak_Current=max_IGBT_peak_Current,
                    max_Diode_RMS_Current=max_Diode_RMS_Current,
                    max_Diode_peak_Current=max_Diode_peak_Current,
                    max_IGBT_temperature=max_IGBT_temperature,
                    max_Diode_temperature=max_Diode_temperature)

                TjI_list.append(T_j_I)
                TjD_list.append(T_j_D)

                if is_last_chunk:
                    sec_idx_list.append(np.full(t.size, sec_idx + 1, dtype=np.int32))
                    time_list.append(t + sec_idx)
                    m_list.append(m)
                    is_I_list.append(is_I)
                    is_D_list.append(is_D)
                    P_sw_I_list.append(P_sw_I)
                    P_sw_D_list.append(P_sw_D)
                    P_con_I_list.append(P_con_I)
                    P_con_D_list.append(P_con_D)
                    P_leg_list.append(P_leg)
                    vs_list.append(vs_inverter)
                    is_inv_list.append(is_inverter)

                del t, m, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D, P_leg, vs_inverter, is_inverter
                del T_j_I, T_j_D

        elif thermal_states == "shared":

            for sec_idx, (Vs_i, Is_i, phi_i, Vdc_i, pf_i) in enumerate(zip(Vs_chunk, Is_chunk, phi_chunk, Vdc_chunk, pf_chunk),start=chunk_start):

                t, m, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D, P_leg, T_j_I, T_j_D, vs_inverter, is_inverter,Tbr_I,Tbr_D,Tbr_p,Tbr_s =\
                    Electro_thermal_behavior_class.Electro_thermal_behavior_shared_thermal_state(dt=dt,         # Input = Float
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
                                                                                                 alpha_s = alpha_s, # Input = Array
                                                                                                 r_I=r_I,           # Input = Array
                                                                                                 r_D=r_D,           # Input = Array
                                                                                                 r_paste=r_paste,   # Input = Array
                                                                                                 r_sink=r_sink,     # Input = Array
                                                                                                 Tamb=Tamb,         # Input = Array
                                                                                                 Tbr_I=Tbr_I,       # Input = Array
                                                                                                 Tbr_D=Tbr_D,       # Input = Array
                                                                                                 Tbr_p=Tbr_p,       # Input = Array
                                                                                                 Tbr_s=Tbr_s)       # Input = Array

                Calculation_functions_class.check_igbt_diode_limits(
                    is_I=is_I, is_D=is_D, T_j_I=T_j_I, T_j_D=T_j_D,
                    max_IGBT_RMS_Current=max_IGBT_RMS_Current,
                    max_IGBT_peak_Current=max_IGBT_peak_Current,
                    max_Diode_RMS_Current=max_Diode_RMS_Current,
                    max_Diode_peak_Current=max_Diode_peak_Current,
                    max_IGBT_temperature=max_IGBT_temperature,
                    max_Diode_temperature=max_Diode_temperature)

                TjI_list.append(T_j_I)
                TjD_list.append(T_j_D)

                if is_last_chunk:
                    sec_idx_list.append(np.full(t.size, sec_idx + 1, dtype=np.int32))
                    time_list.append(t + sec_idx)
                    m_list.append(m)
                    is_I_list.append(is_I)
                    is_D_list.append(is_D)
                    P_sw_I_list.append(P_sw_I)
                    P_sw_D_list.append(P_sw_D)
                    P_con_I_list.append(P_con_I)
                    P_con_D_list.append(P_con_D)
                    P_leg_list.append(P_leg)
                    vs_list.append(vs_inverter)
                    is_inv_list.append(is_inverter)

                del t, m, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D, P_leg, vs_inverter, is_inverter
                del T_j_I, T_j_D

        TjI_all = Calculation_functions_class.cat(TjI_list)
        TjD_all = Calculation_functions_class.cat(TjD_list)

        TjI_mean, TjI_delta, t_cycle_heat_I, time_period_df2 = Calculation_functions_class.window_stats(temp = TjI_all, time_window = 0.02,steps_per_sec=int(1/dt), pf = pf_chunk)
        TjD_mean, TjD_delta, t_cycle_heat_D, _               = Calculation_functions_class.window_stats(temp = TjD_all, time_window = 0.02,steps_per_sec=int(1/dt), pf = pf_chunk)

        if is_last_chunk:
            df_1 = pd.DataFrame({
                "sec_idx":     Calculation_functions_class.cat(sec_idx_list),
                "time_s":      Calculation_functions_class.cat(time_list),
                "m":           Calculation_functions_class.cat(m_list),
                "is_I":        Calculation_functions_class.cat(is_I_list),
                "is_D":        Calculation_functions_class.cat(is_D_list),
                "P_sw_I":      Calculation_functions_class.cat(P_sw_I_list),
                "P_sw_D":      Calculation_functions_class.cat(P_sw_D_list),
                "P_con_I":     Calculation_functions_class.cat(P_con_I_list),
                "P_con_D":     Calculation_functions_class.cat(P_con_D_list),
                "P_leg_all":   Calculation_functions_class.cat(P_leg_list),
                "vs_inverter": Calculation_functions_class.cat(vs_list),
                "is_inverter": Calculation_functions_class.cat(is_inv_list),
                "TjI_all":     Calculation_functions_class.cat(TjI_list),
                "TjD_all":     Calculation_functions_class.cat(TjD_list)})

            df_1.to_parquet(f"{df1_dir}/df_1_{chunk_no}.parquet", engine="pyarrow", compression="zstd")

            del sec_idx_list, time_list, m_list, is_I_list, is_D_list
            del P_sw_I_list, P_sw_D_list, P_con_I_list, P_con_D_list, P_leg_list
            del TjI_list, TjD_list, vs_list, is_inv_list
            del df_1

        del TjI_all, TjD_all
        del Vs_chunk, Is_chunk, phi_chunk, Vdc_chunk, pf_chunk

        Calculation_functions_class.free_ram_now()

        from Electro_thermal_behavior_file import _build_kernel_one_second_cached
        print(_build_kernel_one_second_cached.cache_info())

        Nf_I = Calculation_functions_class.Cycles_to_failure(A=A,
                                                             alpha=alpha,
                                                             beta1=beta1,
                                                             beta0=beta0,
                                                             C=C,
                                                             gamma=gamma,
                                                             fd=fI,
                                                             Ea=Ea,
                                                             k_b=k_b,
                                                             Tj_mean=TjI_mean,
                                                             delta_Tj=TjI_delta,
                                                             t_cycle_heat=t_cycle_heat_I,
                                                             ar=ar )

        Nf_D = Calculation_functions_class.Cycles_to_failure(A=A,
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
                                                             ar=ar )

        chunk_seconds = chunk_end - chunk_start
        time_period_df2 = np.arange(chunk_start, chunk_end, dtype=np.int64)

        TjI_mean = TjI_mean.reshape(-1, f).mean(axis=1)
        TjD_mean = TjD_mean.reshape(-1, f).mean(axis=1)
        TjI_delta = TjI_delta.reshape(-1, f).mean(axis=1)
        TjD_delta = TjD_delta.reshape(-1, f).mean(axis=1)
        Nf_I = Nf_I.reshape(-1, f).mean(axis=1)
        Nf_D = Nf_D.reshape(-1, f).mean(axis=1)
        t_cycle_heat_I = t_cycle_heat_I.reshape(-1, f).mean(axis=1)
        t_cycle_heat_D = t_cycle_heat_D.reshape(-1, f).mean(axis=1)




        df_2 = pd.DataFrame({
            "time_period_df2":time_period_df2,
            "TjI_mean": TjI_mean,
            "TjD_mean": TjD_mean,
            "TjI_delta": TjI_delta,
            "TjD_delta": TjD_delta,
            "Nf_I": Nf_I,
            "Nf_D": Nf_D,
            "t_cycle_heat_I":t_cycle_heat_I,
            "t_cycle_heat_D":t_cycle_heat_D})

        for col in ["time_period_df2","TjI_mean", "TjD_mean", "TjI_delta", "TjD_delta", "Nf_I", "Nf_D","t_cycle_heat_I","t_cycle_heat_D"]:
            df_2[col] = df_2[col].astype("float32")

        #df_2["time_period_df2"] = df_2["time_period_df2"].astype("int32 ")  # or int32 if it’s always integer seconds

        df_2.to_parquet(os.path.join(df2_dir, f"df_2_{chunk_no:04d}.parquet"), engine="pyarrow",compression="zstd",compression_level=7,use_dictionary=True)

        del time_period_df2 , TjI_mean, TjD_mean, TjI_delta , TjD_delta , Nf_I , Nf_D , t_cycle_heat_I , t_cycle_heat_D
        del df_2

        Calculation_functions_class.free_ram_now()

        end_time_chunk = time.time()
        print("Execution time this chunk:", end_time_chunk - start_time_chunk, "seconds")

    df2_files = Calculation_functions_class.find_sorted_files(df2_dir, "df_2")
    Calculation_functions_class.merge_parquet_files(df2_files, os.path.join(Location_dataframes, "df_2.parquet"))

    df_2 = pd.read_parquet(os.path.join(Location_dataframes, "df_2.parquet"), engine="pyarrow")

    Nf_I   = df_2["Nf_I"    ].to_numpy()
    Nf_D   = df_2["Nf_D"    ].to_numpy()
    TjI_mean = df_2["TjI_mean"].to_numpy()
    TjD_mean = df_2["TjD_mean"].to_numpy()

    Life_I = Calculation_functions_class.Lifecycle_calculation_acceleration_factor(Nf = np.repeat(Nf_I, f), pf = pf, Component_max_lifetime = IGBT_max_lifetime)
    Life_D = Calculation_functions_class.Lifecycle_calculation_acceleration_factor(Nf = np.repeat(Nf_D, f), pf = pf, Component_max_lifetime = Diode_max_lifetime)
    Life_switch = min(Life_I, Life_D)

    print('Life_I',Life_I)
    print('Life_D',Life_D)

    df_2.loc[df_2.index[0], ["Life_I", "Life_D", "Life_switch", "dt"]] = [float(Life_I), float(Life_D), float(Life_switch), float(dt)]

    '################################################################################################################################################################'
    'Monte carlo-based reliability assessment'
    '################################################################################################################################################################'

    #----------------------------------------#
    # Calculating delta T from mean T mean and heat cycle values
    #----------------------------------------#

    number_of_yearly_cycles ,Yearly_life_consumption_I, Tj_mean_float_I, delta_Tj_float_I,t_cycle_float = Calculation_functions_class.delta_t_calculations(A = A,                 # Input = float
                                                                                                                                                           alpha = alpha,         # Input = float
                                                                                                                                                           beta1 = beta1,         # Input = float
                                                                                                                                                           beta0 = beta0,         # Input = float
                                                                                                                                                           C = C,                 # Input = float
                                                                                                                                                           gamma = gamma,         # Input = float
                                                                                                                                                           fd = fI,               # Input = float
                                                                                                                                                           Ea = Ea,               # Input = float
                                                                                                                                                           k_b = k_b,             # Input = float
                                                                                                                                                           Tj_mean = TjI_mean,    # Input = array
                                                                                                                                                           t_cycle_float = t_cycle_heat_my_value, # Input = float
                                                                                                                                                           ar = ar,               # Input = float
                                                                                                                                                           Nf = Nf_I,             # Input = float
                                                                                                                                                           pf = pf,               # Input = float
                                                                                                                                                           Life = Life_I,         # Input = float
                                                                                                                                                           f =f)                  # Input = float


    _, Yearly_life_consumption_D, Tj_mean_float_D, delta_Tj_float_D,_ = Calculation_functions_class.delta_t_calculations(A = A,             # Input = float
                                                                                                                         alpha = alpha,         # Input = float
                                                                                                                         beta1 = beta1,         # Input = float
                                                                                                                         beta0 = beta0,         # Input = float
                                                                                                                         C = C,             # Input = float
                                                                                                                         gamma = gamma,         # Input = float
                                                                                                                         fd = fd,            # Input = float
                                                                                                                         Ea = Ea,            # Input = float
                                                                                                                         k_b = k_b,             # Input = float
                                                                                                                         Tj_mean = TjD_mean,    # Input = array
                                                                                                                         t_cycle_float = t_cycle_heat_my_value, # Input = float
                                                                                                                         ar = ar,               # Input = float
                                                                                                                         Nf = Nf_D,             # Input = float
                                                                                                                         pf = pf,               # Input = float
                                                                                                                         Life = Life_D ,        # Input = float
                                                                                                                         f=f)                   # Input = float
    #----------------------------------------#
    # Normal distribution of every variable to calculate the variability of N_f
    #----------------------------------------#

    A_normal_distribution                = Calculation_functions_class.variable_input_normal_distribution(variable = A, normal_distribution = 0.05, number_of_samples = 10000)
    alpha_normal_distribution            = Calculation_functions_class.variable_input_normal_distribution(variable = alpha, normal_distribution = 0.05, number_of_samples = 10000)
    beta1_normal_distribution            = Calculation_functions_class.variable_input_normal_distribution(variable = beta1, normal_distribution = 0.05, number_of_samples = 10000)
    beta0_normal_distribution            = Calculation_functions_class.variable_input_normal_distribution(variable = beta0, normal_distribution = 0.05, number_of_samples = 10000)
    C_normal_distribution                = Calculation_functions_class.variable_input_normal_distribution(variable = C, normal_distribution = 0.05, number_of_samples = 10000)
    gamma_normal_distribution            = Calculation_functions_class.variable_input_normal_distribution(variable = gamma, normal_distribution = 0.05, number_of_samples = 10000)
    fI_normal_distribution               = Calculation_functions_class.variable_input_normal_distribution(variable = fI, normal_distribution = 0.05, number_of_samples = 10000)
    fd_normal_distribution               = Calculation_functions_class.variable_input_normal_distribution(variable = fd, normal_distribution = 0.05, number_of_samples = 10000)
    Ea_normal_distribution               = Calculation_functions_class.variable_input_normal_distribution(variable = Ea, normal_distribution = 0.05, number_of_samples = 10000)
    ar_normal_distribution               = Calculation_functions_class.variable_input_normal_distribution(variable = ar, normal_distribution = 0.05, number_of_samples = 10000)
    k_b_normal_distribution              = Calculation_functions_class.variable_input_normal_distribution(variable = k_b, normal_distribution = 0, number_of_samples = 10000)

    t_cycle_float_normal_distribution    = Calculation_functions_class.variable_input_normal_distribution(variable = t_cycle_float, normal_distribution = 0.05, number_of_samples = 10000)
    Tj_mean_float_I_normal_distribution  = Calculation_functions_class.variable_input_normal_distribution(variable = Tj_mean_float_I, normal_distribution = 0.05, number_of_samples = 10000)
    delta_Tj_float_I_normal_distribution = Calculation_functions_class.variable_input_normal_distribution(variable = delta_Tj_float_I, normal_distribution = 0.05, number_of_samples = 10000)
    Tj_mean_float_D_normal_distribution  = Calculation_functions_class.variable_input_normal_distribution(variable = Tj_mean_float_D, normal_distribution = 0.05, number_of_samples = 10000)
    delta_Tj_float_D_normal_distribution = Calculation_functions_class.variable_input_normal_distribution(variable = delta_Tj_float_D, normal_distribution = 0.05, number_of_samples = 10000)

    Nf_I_normal_distribution = Calculation_functions_class.Cycles_to_failure(A=A_normal_distribution,
                                                                             alpha=alpha_normal_distribution,
                                                                             beta1=beta1_normal_distribution,
                                                                             beta0=beta0_normal_distribution,
                                                                             C=C_normal_distribution,
                                                                             gamma=gamma_normal_distribution,
                                                                             fd=fI_normal_distribution,
                                                                             Ea=Ea_normal_distribution,
                                                                             k_b=k_b_normal_distribution,
                                                                             Tj_mean=Tj_mean_float_I_normal_distribution,
                                                                             delta_Tj=delta_Tj_float_I_normal_distribution,
                                                                             t_cycle_heat=t_cycle_float_normal_distribution,
                                                                             ar=ar_normal_distribution )

    Nf_D_normal_distribution = Calculation_functions_class.Cycles_to_failure(A=A_normal_distribution,
                                                                             alpha=alpha_normal_distribution,
                                                                             beta1=beta1_normal_distribution,
                                                                             beta0=beta0_normal_distribution,
                                                                             C=C_normal_distribution,
                                                                             gamma=gamma_normal_distribution,
                                                                             fd=fd_normal_distribution,
                                                                             Ea=Ea_normal_distribution,
                                                                             k_b=k_b_normal_distribution,
                                                                             Tj_mean=Tj_mean_float_D_normal_distribution,
                                                                             delta_Tj=delta_Tj_float_D_normal_distribution,
                                                                             t_cycle_heat=t_cycle_float_normal_distribution,
                                                                             ar=ar_normal_distribution )

    Life_period_I_normal_distribution = Calculation_functions_class.Lifecycle_normal_distribution_calculation_acceleration_factor(Nf=Nf_I_normal_distribution, f=f, Component_max_lifetime=IGBT_max_lifetime)
    Life_period_D_normal_distribution = Calculation_functions_class.Lifecycle_normal_distribution_calculation_acceleration_factor(Nf=Nf_D_normal_distribution, f=f, Component_max_lifetime=Diode_max_lifetime)
    Life_period_switch_normal_distribution = np.minimum(Life_period_I_normal_distribution,Life_period_D_normal_distribution)

    end_time = time.time()
    print("Execution time all code:", end_time - start_time, "seconds")
    Execution_time = end_time - start_time

    '################################################################################################################################################################'
    'Plotting Values and saving dataframe'
    '################################################################################################################################################################'

    #----------------------------------------#
    # Saving dataframes
    #----------------------------------------#

    # usage
    df_1 = Calculation_functions_class.load_latest_df("df_1", Location_dataframes)


    df_3 = pd.DataFrame({
        "S":S,
        "P": P,
        "Q": Q,
        "phi": phi,
        "pf": pf,
        "Vs": Vs,
        "Is":Is,
        "V_dc": V_dc}).astype("float32")

    df_3.loc[df_3.index[0], ["N_parallel"]] = [float(N_parallel)]
    df_3.loc[df_3.index[0], ["Execution_time"]] = [float(Execution_time)]


    df_4 = pd.DataFrame({
        "A_normal_distribution": A_normal_distribution,
        "alpha_normal_distribution": alpha_normal_distribution,
        "beta1_normal_distribution": beta1_normal_distribution,
        "beta0_normal_distribution": beta0_normal_distribution,
        "C_normal_distribution": C_normal_distribution,
        "gamma_normal_distribution": gamma_normal_distribution,
        "fI_normal_distribution": fI_normal_distribution,
        "fd_normal_distribution": fd_normal_distribution,
        "Ea_normal_distribution": Ea_normal_distribution,
        "ar_normal_distribution": ar_normal_distribution,
        "k_b_normal_distribution": k_b_normal_distribution,
        "t_cycle_float_normal_distribution": t_cycle_float_normal_distribution,
        "Tj_mean_float_I_normal_distribution": Tj_mean_float_I_normal_distribution,
        "delta_Tj_float_I_normal_distribution": delta_Tj_float_I_normal_distribution,
        "Tj_mean_float_D_normal_distribution": Tj_mean_float_D_normal_distribution,
        "delta_Tj_float_D_normal_distribution": delta_Tj_float_D_normal_distribution,

        "Nf_I_normal_distribution":Nf_I_normal_distribution,
        "Nf_D_normal_distribution":Nf_D_normal_distribution,
        "Life_period_I_normal_distribution":Life_period_I_normal_distribution,
        "Life_period_D_normal_distribution":Life_period_D_normal_distribution,
        "Life_period_switch_normal_distribution":Life_period_switch_normal_distribution,
        "inverter_phases":inverter_phases})


    if saving_dataframes == True:
        save_dataframes(df_1 = df_1, df_2 = df_2, df_3 = df_3, df_4 = df_4, Location_dataframes="dataframe_files",timestamp=timestamp)

    if plotting_values == True:
        Plotting_class( df_1 = df_1, df_2 = df_2, df_3 = df_3, df_4 = df_4, Location_plots = "Figures",timestamp=timestamp)



