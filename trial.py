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

df_1 = pd.read_parquet(("dataframe_files/main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51/df_1.parquet"), engine="pyarrow")
df_2 = pd.read_parquet(("dataframe_files/main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51/df_2.parquet"), engine="pyarrow")
df_3 = pd.read_parquet(("dataframe_files/main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51/df_3.parquet"), engine="pyarrow")
df_4 = pd.read_parquet(("dataframe_files/main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51/df_4.parquet"), engine="pyarrow")




Plotting_class(df_1=df_1, df_2=df_2, df_3=df_3, df_4=df_4, Location_plots="Figures", timestamp="main_2_synPRO_el_family_main_2_1_sec_2025-10-10_12-05-51")
