import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(f"Load_profiles/Load_profiles_original/synPRO_el_family.dat", comment="#", sep=";")
df.index = pd.to_datetime(df['unixtimestamp'], unit='s', utc=True)
df = df[['P_el']]

df = df.resample('s').interpolate(method='linear')

df.index = pd.date_range(start='2022-01-01 00:00:00+00:00', periods=len(df), freq='s')

full_index = pd.date_range(start='2022-01-01 00:00:00+00:00', end='2022-12-31 23:59:59+00:00', freq='s')

df = df.reindex(full_index, method='ffill')

max_value = df["P_el"].max()
multiplier = 34495/max_value

df["P_el"] = df["P_el"] * multiplier

df["P_el"] = (df["P_el"] * 3).clip(upper=34495)

print(df)








# --- main_1: average day profile (stack all days, average second-of-day) ---
# seconds since midnight for each timestamp (0..86399)
sod = (df.index.tz_convert('UTC') - df.index.tz_convert('UTC').normalize()).seconds

# average across all days for each second-of-day
avg_profile = df.groupby(sod)['P_el'].mean()

# give it a time-of-day index for a single reference day
tod_index = pd.date_range('2022-01-01 00:00:00+00:00', periods=86400, freq='s')
main_1 = avg_profile.to_frame().set_index(tod_index)
main_1.index.name = 'time_of_day'  # optional

# --- main_2: same as df ---
main_2_1_sec = df.copy()

# --- main_3: per-day average replicated across each second of that day ---
main_2_15_min = df.copy()

group = df.index.floor('15min') # grouping by 15 min
#group = df.index.floor('h') # group per hour
#group = df.index.normalize()  # group per day
#group = ((df.index.normalize() - df.index.normalize()[0]).days // 5) # group more than 1 day

main_2_15_min['P_el'] = df['P_el'].groupby(group).transform('mean')


main_1.to_parquet(f"Load_profiles/synPRO_el_family_main_1.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)
main_2_1_sec.to_parquet(f"Load_profiles/synPRO_el_family_main_2_1_sec.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)
main_2_15_min.to_parquet(f"Load_profiles/synPRO_el_family_main_2_15_min.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)



