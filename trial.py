import os
import pandas as pd
import matplotlib.pyplot as plt

# Load both dataframes
df_4_main_1 = pd.read_parquet(os.path.join("dataframe_files/main_1", "df_4.parquet"), engine="pyarrow")
df_4_main_2 = pd.read_parquet(os.path.join("dataframe_files/main_2", "df_4.parquet"), engine="pyarrow")

# Extract the four series
Tj_mean_1 = df_4_main_1["Tj_mean_float_I_normal_distribution"]
delta_Tj_1 = df_4_main_1["Nf_I_normal_distribution"]

Tj_mean_2 = df_4_main_2["Tj_mean_float_I_normal_distribution"]
delta_Tj_2 = df_4_main_2["Nf_I_normal_distribution"]

# Plot all on one figure
plt.figure(figsize=(10,6))
#plt.plot(Tj_mean_1-275.15, label="Tj_mean (main_1)")
plt.plot(delta_Tj_1, label="delta_Tj (main_1)")
#plt.plot(Tj_mean_2-275.15, label="Tj_mean (main_2)")
plt.plot(delta_Tj_2, label="delta_Tj (main_2)")

plt.title("Comparison of Tj_mean and delta_Tj (main_1 vs main_2)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
