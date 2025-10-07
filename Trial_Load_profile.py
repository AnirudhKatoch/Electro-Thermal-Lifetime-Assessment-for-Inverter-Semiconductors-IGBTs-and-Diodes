import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_parquet(f"Load_profiles/synPRO_el_family_1_sec_1_year.parquet", engine="pyarrow")


P_el = np.array(df["P_el"])

plt.plot(P_el)
plt.show()


