import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df_1 = pd.read_csv(f"Load_profiles/Load_profiles_original/synPRO_el_family.dat", comment="#", sep=";")
df_1.index = pd.to_datetime(df_1['unixtimestamp'], unit='s', utc=True)

df_2 = pd.read_csv(f"Load_profiles/Load_profiles_original/synPRO_el_2_fulltime_employees.dat", comment="#", sep=";")
df_2.index = pd.to_datetime(df_2['unixtimestamp'], unit='s', utc=True)

df_3 = pd.read_csv(f"Load_profiles/Load_profiles_original/synPRO_el_2_persons_over65.dat", comment="#", sep=";")
df_3.index = pd.to_datetime(df_3['unixtimestamp'], unit='s', utc=True)

df_4 = pd.read_csv(f"Load_profiles/Load_profiles_original/synPRO_el_single_person_under30.dat", comment="#", sep=";")
df_4.index = pd.to_datetime(df_4['unixtimestamp'], unit='s', utc=True)

# Drop unnecessary columns
cols_to_drop = ['YYYYMMDD', 'hhmmss', 'unixtimestamp']
df_1 = df_1.drop(columns=cols_to_drop)
df_2 = df_2.drop(columns=cols_to_drop)
df_3 = df_3.drop(columns=cols_to_drop)
df_4 = df_4.drop(columns=cols_to_drop)

df = pd.DataFrame(index=df_1.index)
df['P'] = df_1['P_el'] + df_2['P_el'] + df_3['P_el'] + df_4['P_el']

df = df.resample('s').interpolate(method='linear')

df.index = pd.date_range(start='2022-01-01 00:00:00+00:00', periods=len(df), freq='s')

full_index = pd.date_range(start='2022-01-01 00:00:00+00:00', end='2022-12-31 23:59:59+00:00', freq='s')

df = df.reindex(full_index, method='ffill')

max_value = df["P"].max()

N_parallel = 50
Rated_power = 34450 * N_parallel

Factor_increase = 2.5

multiplier = Rated_power/max_value
df["P"] = (df["P"] * multiplier * Factor_increase ).clip(upper=Rated_power)

S_rated = Rated_power

# --- compute per-second capability first ---
P = df["P"].to_numpy(dtype=float)
Q_max = np.sqrt(np.maximum(S_rated**2 - P**2, 0.0))  # array same length as df

df["Q_max"] = Q_max

# define 15-minute groups for dispatch
df["group"] = df.index.floor('15min')

rng = np.random.default_rng(seed=42)

# we'll fill this array
Q_blockwise = np.zeros_like(Q_max)

for g, block_idx in df.groupby("group").groups.items():
    # block_idx is an index of TIMESTAMPS (labels), not integer positions

    # safe kvar capability for the entire 15-min block: pick the minimum capability
    #block_Qcap = df.loc[block_idx, "Q_max"].min()

    safety = 0.90
    block_Qcap = df.loc[block_idx, "Q_max"].min() * safety

    # random magnitude within [0, block_Qcap]
    mag = rng.random() * block_Qcap

    # random sign (+ absorbing inductive / - capacitive, or whatever convention you're using)
    sign = rng.choice([-1.0, 1.0])

    # base kvar setpoint for this 15-min interval
    q_set = mag * sign

    # small jitter (±2% of |q_set|)
    noise = rng.normal(
        loc=0.0,
        scale=0.02 * abs(q_set),
        size=len(block_idx)
    )

    q_values_block = q_set + noise

    # clip against instantaneous capability at each second in this block
    q_cap_this_block = df.loc[block_idx, "Q_max"].to_numpy()
    q_values_block = np.clip(q_values_block,
                             -q_cap_this_block,
                              q_cap_this_block)

    # assign back: we can't do Q_blockwise[block_idx] because block_idx are timestamps,
    # but we CAN assign via df.loc[...] and then later read it out.
    df.loc[block_idx, "Q"] = q_values_block

# now that df["Q"] exists, compute S and pf
Q = df["Q"].to_numpy()
S = np.hypot(P, Q)

pf_magnitude = np.divide(P, S, out=np.zeros_like(P), where=S > 0)

def sign_q_func(Q_arr):
    sign_q = np.where(np.abs(Q_arr) > 1e-9, np.sign(Q_arr), 1.0)
    return sign_q

pf = sign_q_func(Q) * pf_magnitude

df["S"]  = S
df["pf"] = pf

# cleanup helper cols
df = df.drop(columns=["Q_max", "group"])

def load_profile_builder(df,Factor_increase):

    '''

    def sign_q_func(Q):
        sign_q = np.where(np.abs(Q) > 1e-9, np.sign(Q), 1.0)
        return sign_q

    # --- main_1: average day profile (stack all days, average second-of-day) ---
    # seconds since midnight for each timestamp (0..86399)
    sod = (df.index.tz_convert('UTC') - df.index.tz_convert('UTC').normalize()).seconds

    # average across all days for each second-of-day
    avg_P = df.groupby(sod)["P"].mean()
    avg_Q = df.groupby(sod)["Q"].mean()

    # give it a time-of-day index for a single reference day
    avg_S = np.hypot(avg_P.to_numpy(), avg_Q.to_numpy())
    avg_pf_signed = np.divide(avg_P.to_numpy(), avg_S, out=np.zeros_like(avg_S), where=avg_S > 0)

    avg_pf_signed = sign_q_func(avg_Q.to_numpy()) * np.abs(avg_pf_signed)

    # build main_1 dataframe with time-of-day index
    tod_index = pd.date_range('2022-01-01 00:00:00+00:00', periods=86400, freq='s')
    main_1 = pd.DataFrame({
        "P": avg_P.values,
        "Q": avg_Q.values,
        "pf": avg_pf_signed,
    }, index=tod_index)

    '''

    # --- MAIN_2: same as df (1-second) ---
    main_2_1_sec = df[["P", "Q", "pf"]].copy()

    '''

    # --- main_3: per-day average replicated across each second of that day ---

    group = df.index.floor('15min') # grouping by 15 min
    #group = df.index.floor('h') # group per hour
    #group = df.index.normalize()  # group per day
    #group = ((df.index.normalize() - df.index.normalize()[0]).days // 5) # group more than 1 day

    # compute block means of P and Q
    block_P = df["P"].groupby(group).transform('mean')
    block_Q = df["Q"].groupby(group).transform('mean')

    # recompute pf from block means
    block_S = np.hypot(block_P.to_numpy(), block_Q.to_numpy())
    block_pf = np.divide(block_P.to_numpy(), block_S, out=np.zeros_like(block_S), where=block_S > 0)
    block_pf = sign_q_func(block_Q.to_numpy()) * np.abs(block_pf)

    main_2_15_min = pd.DataFrame({
        "P": block_P.values,
        "Q": block_Q.values,
        "pf": block_pf,
    }, index=df.index)
    '''

    #main_1.to_parquet(f"Load_profiles/alle_main_2_1_{Factor_increase}.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)
    main_2_1_sec.to_parquet(f"Load_profiles/alle_main_2_1_sec_inverter_{Factor_increase}.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)
    #main_2_15_min.to_parquet(f"Load_profiles/alle_main_2_15_min_{Factor_increase}.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)

load_profile_builder(df,Factor_increase)
