import numpy as np
import pandas as pd
from Input_parameters_file import Input_parameters_class
from Calculation_functions_file import Calculation_functions_class
from Electro_thermal_behavior_file import Electro_thermal_behavior_class
from Plotting_and_saving_dataframe_file import Plotting_and_saving_dataframe_class
import time

start_time = time.time()

Input_parameters = Input_parameters_class()

'################################################################################################################################################################'
'Input Parameters'
'################################################################################################################################################################'

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

S,Is,phi,P,Q,Vs = Calculation_functions_class.compute_power_flow_from_pf(P=P, Q=Q, V_dc = V_dc, pf=pf, Vs=Vs)  # Inverter Power Flow

#----------------------------------------#
# Temperature calculations
#----------------------------------------#

Tbr_I = np.zeros_like(r_I, dtype=float)     # [K] Temperature rise contributions of each Foster RC branch for IGBT junction → case.
Tbr_D = np.zeros_like(r_D, dtype=float)     # [K] Temperature rise contributions of each Foster RC branch for diode junction → case.
Tbr_p = np.zeros_like(r_paste, dtype=float) # [K] Temperature rise across the thermal paste (case → heatsink interface).
Tbr_s = np.zeros_like(r_sink, dtype=float)  # [K] Temperature rise contributions of each Foster RC branch for the heatsink (sink → ambient).

P_leg_all = []
TjI_all   = []
TjD_all   = []
all_rows = []

for sec_idx, (Vs_i, Is_i, phi_i, Vdc_i, pf_i) in enumerate(zip(Vs, Is, phi, V_dc, pf)):

    t, m, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D, P_leg, T_j_I, T_j_D, vs_inverter, is_inverter,Tbr_I,Tbr_D,Tbr_p,Tbr_s = Electro_thermal_behavior_class.Electro_thermal_behavior_shared_thermal_state(dt=dt,             # Input = Float
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
        max_Diode_temperature=max_Diode_temperature,
        sec_idx=sec_idx+1
    )

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

    #print(sec_idx+1)

P_leg_all = np.concatenate(P_leg_all)  # [W] total leg power over full run
TjI_all   = np.concatenate(TjI_all)    # [K] IGBT junction temp over full run
TjD_all   = np.concatenate(TjD_all)    # [K] Diode junction temp over full run

TjI_mean, TjI_delta, t_cycle_heat_I, time_period_df2 = Calculation_functions_class.window_stats(temp = TjI_all, time_window = 0.02,steps_per_sec=int(1/dt), pf = pf)
TjD_mean, TjD_delta, t_cycle_heat_D, _               = Calculation_functions_class.window_stats(temp = TjD_all, time_window = 0.02,steps_per_sec=int(1/dt), pf = pf)




#----------------------------------------#
# Life cycle calculations (Its use simple model if that does not work it just uses static model)
#----------------------------------------#


#----------------------------------------#
# IGBT
#----------------------------------------#

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

Life_I = Calculation_functions_class.Lifecycle_calculation(Nf_I, pf)


if Life_I > IGBT_max_lifetime:

    _, _, _, dT_equiv_I, t_cycle_float_equiv_I = Calculation_functions_class.delta_t_calculations(A=A,
                                                                             alpha=alpha,
                                                                             beta1=beta1,
                                                                             beta0=beta0,
                                                                             C=C,
                                                                             gamma=gamma,
                                                                             fd=fI,
                                                                             Ea=Ea,
                                                                             k_b=k_b,
                                                                             Tj_mean = TjI_mean,
                                                                             t_cycle_float = t_cycle_heat_my_value,
                                                                             ar=ar,
                                                                             Nf = np.ones_like(TjI_mean),
                                                                             pf = pf,
                                                                             Life = IGBT_max_lifetime)

    Nf_prime_I = Calculation_functions_class.Cycles_to_failure(A=A,
                                                               alpha=alpha,
                                                               beta1=beta1,
                                                               beta0=beta0,
                                                               C=C,
                                                               gamma=gamma,
                                                               fd=fI,
                                                               Ea=Ea,
                                                               k_b=k_b,
                                                               Tj_mean=np.array([float(np.mean(TjI_mean))]),
                                                               delta_Tj=np.array([float(dT_equiv_I)]),
                                                               t_cycle_heat=t_cycle_heat_my_value,
                                                               ar=ar)

    LC_year_I = (len(TjI_mean) / len(pf)) * 3600 * 24 * 365 / Nf_prime_I
    Life_I = (1.0 / LC_year_I) if LC_year_I > 0 else float("inf")
    if isinstance(Life_I, (np.ndarray,)):
        Life_I = Life_I.item()

#----------------------------------------#
# Diode
#----------------------------------------#

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

Life_D = Calculation_functions_class.Lifecycle_calculation(Nf_D,pf)

if Life_D > Diode_max_lifetime:

    _, _, _, dT_equiv_D, _ = Calculation_functions_class.delta_t_calculations(A=A,
                                                                              alpha=alpha,
                                                                              beta1=beta1,
                                                                              beta0=beta0,
                                                                              C=C,
                                                                              gamma=gamma,
                                                                              fd=fd,
                                                                              Ea=Ea,
                                                                              k_b=k_b,
                                                                              Tj_mean=TjD_mean,
                                                                              t_cycle_float=t_cycle_heat_my_value,
                                                                              ar=ar,
                                                                              Nf=np.ones_like(TjD_mean),
                                                                              pf=pf,
                                                                              Life=Diode_max_lifetime)

    Nf_prime_D = Calculation_functions_class.Cycles_to_failure(A=A,
                                                               alpha=alpha,
                                                               beta1=beta1,
                                                               beta0=beta0,
                                                               C=C,
                                                               gamma=gamma,
                                                               fd=fd,
                                                               Ea=Ea,
                                                               k_b=k_b,
                                                               Tj_mean=np.array([float(np.mean(TjD_mean))]),
                                                               delta_Tj=np.array([float(dT_equiv_D)]),
                                                               t_cycle_heat=t_cycle_heat_my_value,
                                                               ar=ar)

    LC_year_D = (len(TjD_mean) / len(pf)) * 3600 * 24 * 365 / Nf_prime_D
    Life_D = (1.0 / LC_year_D) if LC_year_D > 0 else float("inf")
    if isinstance(Life_D, (np.ndarray,)):
        Life_D = Life_D.item()

Life_switch = min(Life_I, Life_D)

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
                                                                                                                                                       Life = Life_I )        # Input = float


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
                                                                                                                                 Life = Life_D )        # Input = float


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

Life_period_I_normal_distribution = Nf_I_normal_distribution / (3600 * 24 * 365 * (len(Nf_I)/len(pf)))
Life_period_D_normal_distribution = Nf_D_normal_distribution / (3600 * 24 * 365 * (len(Nf_I)/len(pf)))
Life_period_switch_normal_distribution = np.minimum(Life_period_I_normal_distribution,Life_period_D_normal_distribution)

end_time = time.time()
print("Execution time:", end_time - start_time, "seconds")

'################################################################################################################################################################'
'Plotting Values and saving dataframe'
'################################################################################################################################################################'


#----------------------------------------#
# Saving dataframes
#----------------------------------------#

df_1 = pd.concat(all_rows, ignore_index=True)
df_1["P_leg_all"] = P_leg_all
df_1["TjI_all"]   = TjI_all
df_1["TjD_all"]   = TjD_all

df_2 = pd.DataFrame({
    "time_period_df2":time_period_df2,
    "TjI_mean": TjI_mean,
    "TjD_mean": TjD_mean,
    "TjI_delta": TjI_delta,
    "TjD_delta": TjD_delta,
    "Nf_I": Nf_I,
    "Nf_D": Nf_D,
    "t_cycle_heat_I":t_cycle_heat_I,
    "t_cycle_heat_D":t_cycle_heat_D,
    "Life_I":Life_I,
    "Life_D":Life_D,
    "Life_switch":Life_switch})

df_3 = pd.DataFrame({
    "S":S,
    "P": P,
    "Q": Q,
    "phi": phi,
    "pf": pf,
    "Vs": Vs,
    "Is":Is,
    "V_dc": V_dc})

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
    "Life_period_switch_normal_distribution":Life_period_switch_normal_distribution })

Plotting_and_saving_dataframe_class( df_1 = df_1, df_2 = df_2, df_3 = df_3, df_4 = df_4, Location_plots = "Figures", Location_dataframes = "dataframe_files")



