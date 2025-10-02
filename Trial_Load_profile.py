import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Load_profiles/synPRO_el_family.dat", comment="#", sep=";")
df.index = pd.to_datetime(df['unixtimestamp'], unit='s', utc=True)
df = pd.concat([df.iloc[4:], df.iloc[:4]])
P_el = np.array(df['P_el'])


plt.plot(P_el[:96*5],label="P_el")
plt.show()