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
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
from Calculation_functions_file import Calculation_functions_class

#df_2 = pd.read_parquet(("dataframe_files/main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51/df_2.parquet"), engine="pyarrow")
#df_3 = pd.read_parquet(("dataframe_files/main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51/df_3.parquet"), engine="pyarrow")
#df_4 = pd.read_parquet(("dataframe_files/main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51/df_4.parquet"), engine="pyarrow")


#Plotting_class(df_1=df_1, df_2=df_2, df_3=df_3, df_4=df_4, Location_plots="Figures", timestamp="main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51")

df = pd.read_parquet(("Load_profiles/alle_main_2_1_sec_inverter_2.5.parquet"), engine="pyarrow")

print(df)

# --- constants ---
S_LIMIT = 34450*50  # VA = 1.7225 MVA

# --- 1. recompute S and pf from P and Q ---
P = df["P"].to_numpy(dtype=float)
Q = df["Q"].to_numpy(dtype=float)

# apparent power magnitude
S_calc = np.hypot(P, Q)  # sqrt(P^2 + Q^2)

# signed pf = P / S with sign determined by Q
pf_mag = np.divide(P, S_calc, out=np.zeros_like(P), where=S_calc > 0)

def sign_q_func(Q_arr):
    # if Q ~ 0, assume +1 sign so pf stays positive
    sign_q = np.where(np.abs(Q_arr) > 1e-9, np.sign(Q_arr), 1.0)
    return sign_q

pf_signed_calc = sign_q_func(Q) * pf_mag

# --- 2. CHECK #1: S must never exceed limit ---
viol_S = S_calc > S_LIMIT
num_viol_S = np.count_nonzero(viol_S)

print(f"[S limit check]")
print(f"  Max S_calc = {S_calc.max():.2f} VA")
print(f"  Limit      = {S_LIMIT:.2f} VA")
print(f"  Violations = {num_viol_S} points "
      f"({100.0 * num_viol_S / len(df):.6f} % of samples)")
print()

# --- 3. CHECK #2: df['pf'] should match recomputed pf_signed_calc ---
if "pf" in df.columns:
    pf_in = df["pf"].to_numpy(dtype=float)

    # absolute difference
    pf_diff = np.abs(pf_in - pf_signed_calc)

    # choose a numerical tolerance for float math
    tol = 1e-6
    viol_pf = pf_diff > tol
    num_viol_pf = np.count_nonzero(viol_pf)

    print(f"[pf consistency check]")
    print(f"  pf_in range      = [{pf_in.min():.6f}, {pf_in.max():.6f}]")
    print(f"  pf_calc range    = [{pf_signed_calc.min():.6f}, {pf_signed_calc.max():.6f}]")
    print(f"  Max |Δpf|        = {pf_diff.max():.6e}")
    print(f"  Violations >{tol} = {num_viol_pf} points "
          f"({100.0 * num_viol_pf / len(df):.6f} % of samples)")
    print()
else:
    print("Column 'pf' not found in df; skipping pf check.\n")


# --- 4. OPTIONAL sanity prints ---
print(f"[P,Q sanity]")
print(f"  P: min={P.min():.2f} W, max={P.max():.2f} W")
print(f"  Q: min={Q.min():.2f} var, max={Q.max():.2f} var")
print(f"  S: min={S_calc.min():.2f} VA, max={S_calc.max():.2f} VA")
print()

# --- 5. EXTRA CHECKS that help catch modeling issues ---

# A. Is |Q| ever > sqrt(S_LIMIT^2 - P^2) ? That would be physically impossible.
Q_cap_allowed = np.sqrt(np.maximum(S_LIMIT**2 - P**2, 0.0))
viol_Qcap = np.abs(Q) - Q_cap_allowed > 1e-6  # little tolerance
num_viol_Qcap = np.count_nonzero(viol_Qcap)

print(f"[capability check]")
print(f"  Violations of |Q| <= sqrt(Srated^2 - P^2): {num_viol_Qcap} points "
      f"({100.0 * num_viol_Qcap / len(df):.6f} %)")
print()

# B. Power factor magnitude must be between 0 and 1
pf_mag_in = np.abs(df["pf"].to_numpy(dtype=float)) if "pf" in df.columns else np.abs(pf_signed_calc)
viol_pf_bounds = (pf_mag_in < -1e-6) | (pf_mag_in > 1 + 1e-6)
num_viol_pf_bounds = np.count_nonzero(viol_pf_bounds)

print(f"[pf bounds check]")
print(f"  pf magnitude min={pf_mag_in.min():.6f}, max={pf_mag_in.max():.6f}")
print(f"  pf outside [0,1] = {num_viol_pf_bounds} points")
print()

# C. Look for NaNs/Infs (these can happen at S=0 or from divide-by-zero)
bad_nan = np.isnan(P) | np.isnan(Q)
bad_inf = ~np.isfinite(P) | ~np.isfinite(Q)
print(f"[NaN / Inf check]")
print(f"  NaN in P/Q: {np.count_nonzero(bad_nan)}")
print(f"  non-finite in P/Q: {np.count_nonzero(bad_inf)}")
print()

# (optional) if you want to add the recomputed S and pf into df for inspection:
df["S_check"]  = S_calc
df["pf_check"] = pf_signed_calc


# --- Select the first 24 hours ---
start_time = df.index[0]
end_time = start_time + pd.Timedelta(hours=24)
df_24h = df.loc[start_time:end_time]

# --- P & Q plot on one graph ---
fig, ax1 = plt.subplots(figsize=(12, 5))

color_p = 'tab:blue'
color_q = 'tab:orange'

ax1.plot(df_24h.index, df_24h['P'], color=color_p, label='Active Power P [W]')
ax1.set_ylabel('Active Power P [W]', color=color_p)
ax1.tick_params(axis='y', labelcolor=color_p)

# second y-axis for Q
ax2 = ax1.twinx()
ax2.plot(df_24h.index, df_24h['Q'], color=color_q, label='Reactive Power Q [var]', alpha=0.7)
ax2.set_ylabel('Reactive Power Q [var]', color=color_q)
ax2.tick_params(axis='y', labelcolor=color_q)

ax1.set_title("Active (P) and Reactive (Q) Power – First 24 Hours")
ax1.set_xlabel("Time [UTC]")

# optional grid and legend
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- Power factor plot ---
fig, ax = plt.subplots(figsize=(12, 3))

ax.plot(df_24h.index, df_24h['pf'], color='tab:green', label='Power Factor')
ax.set_ylabel('Power Factor [-]')
ax.set_xlabel('Time [UTC]')
ax.set_title('Power Factor – First 24 Hours')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.set_ylim(-1.05, 1.05)
ax.legend()

plt.tight_layout()
plt.show()