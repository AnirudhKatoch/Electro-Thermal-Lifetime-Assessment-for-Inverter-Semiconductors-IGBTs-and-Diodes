import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from datetime import datetime
import os
from Calculation_functions_file import Calculation_functions_class

# -------------------------------------------------
# Global style
# -------------------------------------------------

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 12

class Plotting_and_saving_dataframe_class:

    def __init__(self,df_1,df_2,df_3,df_4,Location_plots,Location_dataframes):
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        Location_dataframes = "dataframe_files"
        Location_dataframes = f"{Location_dataframes}/{timestamp}"
        os.makedirs(Location_dataframes, exist_ok=True)

        df_1.to_parquet(f"{Location_dataframes}/df_1.parquet", engine="pyarrow", compression="zstd")
        df_2.to_parquet(f"{Location_dataframes}/df_2.parquet", engine="pyarrow", compression="zstd")
        df_3.to_parquet(f"{Location_dataframes}/df_3.parquet", engine="pyarrow", compression="zstd")
        df_4.to_parquet(f"{Location_dataframes}/df_4.parquet", engine="pyarrow", compression="zstd")

        Location_plots = f"{Location_plots}/{timestamp}"
        os.makedirs(Location_plots, exist_ok=True)


        # -------------------------------------------------
        # Figure 1: IGBT and Diode mean temperature
        # -------------------------------------------------

        fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))
        ax1.plot(df_2["time_period_df2"], df_2["TjI_mean"] - 273.15, label="IGBT")
        ax1.plot(df_2["time_period_df2"], df_2["TjD_mean"] - 273.15, label="Diode")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title(" IGBT and diode mean temperature")
        ax1.set_xlim(min(df_2["time_period_df2"]), max(df_2["time_period_df2"]))
        ax1.legend(loc="best")
        ax1.grid(True)
        plt.savefig(f"{Location_plots}/1_IGBT_and_Diode_mean_temperature.png")
        plt.close(fig1)


        # -------------------------------------------------
        # Figure 2: IGBT and Diode delta temperature
        # -------------------------------------------------

        fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
        ax2.plot(df_2["time_period_df2"], df_2["TjI_delta"], label="IGBT")
        ax2.plot(df_2["time_period_df2"], df_2["TjD_delta"], label="Diode")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_title(" IGBT and diode temperature variation")
        ax2.set_xlim(min(df_2["time_period_df2"]), max(df_2["time_period_df2"]))
        ax2.legend(loc="best")
        ax2.grid(True)
        plt.savefig(f"{Location_plots}/2_IGBT_and_Diode_delta_temperature.png")
        plt.close(fig2)


        # -------------------------------------------------
        # Figure 3: IGBT and Diode cycles to failure
        # -------------------------------------------------

        fig3, ax3 = plt.subplots(figsize=(6.4, 4.8))
        ax3.plot(df_2["time_period_df2"], df_2["Nf_I"], label="IGBT")
        ax3.plot(df_2["time_period_df2"], df_2["Nf_D"], label="Diode")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Cycles to Failure")
        ax3.set_title("IGBT and Diode cycles to failure")
        ax3.set_xlim(min(df_2["time_period_df2"]), max(df_2["time_period_df2"]))
        ax3.legend()
        ax3.legend(loc="best")
        ax3.grid(True)
        ax3.set_yscale("log")
        plt.savefig(f"{Location_plots}/3_IGBT_and_Diode_cycles_to_failure.png")
        plt.close(fig3)


        # -------------------------------------------------
        # Figure 4: Instantaneous modulation
        # -------------------------------------------------

        fig4, ax4 = plt.subplots(figsize=(6.4, 4.8))
        ax4.plot(df_1["time_s"], df_1["m"], label=" Modulation")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Value")
        ax4.set_title("Modulation (One cycle)")
        ax4.set_xlim(max(df_1["time_s"])-0.04+0.001, max(df_1["time_s"])-0.02+0.001)
        ax4.legend(loc="best")
        ax4.grid(True)
        plt.savefig(f"{Location_plots}/4_Instantaneous_modulation.png")
        plt.close(fig4)


        # -------------------------------------------------
        # Figure 5: Instantaneous IGBT and Diode Current
        # -------------------------------------------------

        fig5, ax5 = plt.subplots(figsize=(6.4, 4.8))
        ax5.plot(df_1["time_s"], df_1["is_I"], label="IGBT")
        ax5.plot(df_1["time_s"], df_1["is_D"], label="Diode")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Current (A)")
        ax5.set_title("IGBT and diode current (One cycle)")
        ax5.set_xlim(max(df_1["time_s"])-0.04+0.001, max(df_1["time_s"])-0.02+0.001)
        ax5.legend(loc="best")
        ax5.grid(True)
        plt.savefig(f"{Location_plots}/5_IGBT_and_Diode_instantaneous_current.png")
        plt.close(fig5)


        # -------------------------------------------------
        # Figure 6: IGBT and Diode instantaneous switching losses
        # -------------------------------------------------

        fig6, ax6 = plt.subplots(figsize=(6.4, 4.8))
        ax6.plot(df_1["time_s"], df_1["P_sw_I"], label="IGBT")
        ax6.plot(df_1["time_s"], df_1["P_sw_D"], label="Diode")
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Power (W)")
        ax6.set_title("IGBT and diode switching losses (One cycle)")
        ax6.set_xlim(max(df_1["time_s"])-0.04+0.001, max(df_1["time_s"])-0.02+0.001)
        ax6.legend(loc="best")
        ax6.grid(True)
        plt.savefig(f"{Location_plots}/6_IGBT_and_Diode_instantaneous_switching_losses.png")
        plt.close(fig6)


        # -------------------------------------------------
        # Figure 7: IGBT and Diode instantaneous conduction losses
        # -------------------------------------------------

        fig7, ax7 = plt.subplots(figsize=(6.4, 4.8))
        ax7.plot(df_1["time_s"], df_1["P_con_I"], label="IGBT")
        ax7.plot(df_1["time_s"], df_1["P_con_D"], label="Diode")
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("Power (W)")
        ax7.set_title("IGBT and Diode conduction losses (One cycle)")
        ax7.set_xlim(max(df_1["time_s"])-0.04+0.001, max(df_1["time_s"])-0.02+0.001)
        ax7.legend(loc="best")
        ax7.grid(True)
        plt.savefig(f"{Location_plots}/7_IGBT_and_Diode_instantaneous_conduction_losses.png")
        plt.close(fig7)


        # -------------------------------------------------
        # Figure 8: Inverter Voltage and Current
        # -------------------------------------------------

        fig8, ax8 = plt.subplots(figsize=(6.4, 4.8))
        ax8.plot(df_1["time_s"], df_1["vs_inverter"], label="Voltage (V)")
        ax8.plot(df_1["time_s"], df_1["is_inverter"], label="Current (A)")
        ax8.set_xlabel("Time (s)")
        ax8.set_ylabel("Value")
        ax8.set_title("Inverter voltage and current (One cycle)")
        ax8.set_xlim(max(df_1["time_s"])-0.04+0.001, max(df_1["time_s"])-0.02+0.001)
        ax8.legend(loc="best")
        ax8.grid(True)
        plt.savefig(f"{Location_plots}/8_Inverter_instantaneous_current_and_voltage.png")
        plt.close(fig8)


        # -------------------------------------------------
        # Figure 9: IGBT and Diode temperature (Last Section)
        # -------------------------------------------------

        fig9, ax9 = plt.subplots(figsize=(6.4, 4.8))
        ax9.plot(df_1["time_s"], df_1["TjI_all"]  - 273.15, label="IGBT")
        ax9.plot(df_1["time_s"], df_1["TjD_all"] - 273.15, label="Diode")
        ax9.set_xlabel("Time (s)")
        ax9.set_ylabel("Temperature (°C)")
        ax9.set_title("IGBT and diode temperature (One cycle)")
        ax9.set_xlim(max(df_1["time_s"])-0.04+0.001, max(df_1["time_s"])-0.02+0.001)
        ax9.legend(loc="best")
        ax9.grid(True)
        plt.savefig(f"{Location_plots}/9_IGBT_and_Diode_instantaneous_temperature_last Section.png")
        plt.close(fig9)


        # -------------------------------------------------
        # Figure 10: IGBT and Diode instantaneous total power losses
        # -------------------------------------------------

        fig10, ax10 = plt.subplots(figsize=(6.4, 4.8))
        ax10.plot(df_1["time_s"], df_1["P_leg_all"], label="Total power losses")
        ax10.set_xlabel("Time (s)")
        ax10.set_ylabel("Power (W)")
        ax10.set_title("IGBT and diode total power losses (One cycle)")
        ax10.set_xlim(max(df_1["time_s"])-0.04+0.001, max(df_1["time_s"])-0.02+0.001)
        ax10.legend(loc="best")
        ax10.grid(True)
        plt.savefig(f"{Location_plots}/10_IGBT_and_Diode_instantaneous_total_power_losses.png")
        plt.close(fig10)


        # -------------------------------------------------
        # Figure 11: IGBT and Diode temperature
        # -------------------------------------------------

        fig11, ax11 = plt.subplots(figsize=(6.4, 4.8))
        ax11.plot(df_1["time_s"], df_1["TjI_all"]  - 273.15, label="IGBT")
        ax11.plot(df_1["time_s"], df_1["TjD_all"] - 273.15, label="Diode")
        ax11.set_xlabel("Time (s)")
        ax11.set_ylabel("Temperature (°C)")
        ax11.set_title("IGBT and diode temperature")
        ax11.set_xlim(min(df_1["time_s"]), max(df_1["time_s"]))
        ax11.legend(loc="best")
        ax11.grid(True)
        plt.savefig(f"{Location_plots}/11_IGBT_and_Diode_instantaneous_temperature_full_time_scale.png")
        plt.close(fig11)


        # -------------------------------------------------
        # Figure 12: Active, reactive and apparent power
        # -------------------------------------------------

        fig12, ax12 = plt.subplots(figsize=(6.4, 4.8))
        ax12.plot(np.arange(0, len(df_3["pf"]), 1), df_3["S"], label="Apparent power", linewidth=5)
        ax12.plot(np.arange(0, len(df_3["pf"]), 1), df_3["P"], label="Active power")
        ax12.plot(np.arange(0, len(df_3["pf"]), 1), df_3["Q"], label="Reactive power")
        ax12.set_xlabel("Time (s)")
        ax12.set_ylabel("Power (W)")
        ax12.set_title("Active, reactive and apparent power")
        ax12.set_xlim(min(np.arange(0, len(df_3["pf"]), 1)), max(np.arange(0, len(df_3["pf"]), 1)))
        ax12.legend(loc="best")
        ax12.grid(True)
        plt.savefig(f"{Location_plots}/12_Active,_reactive_and_apparent_power.png")
        plt.close(fig12)


        # -------------------------------------------------
        # Figure 13: Power Factor
        # -------------------------------------------------

        fig13, ax13 = plt.subplots(figsize=(6.4, 4.8))
        ax13.plot(np.arange(0, len(df_3["pf"]), 1), df_3["pf"], label="Power Factor")
        ax13.set_xlabel("Time (s)")
        ax13.set_ylabel("Value")
        ax13.set_title("Power Factor")
        ax13.set_xlim(min(np.arange(0, len(df_3["pf"]), 1)), max(np.arange(0, len(df_3["pf"]), 1)))
        ax13.legend(loc="best")
        ax13.grid(True)
        plt.savefig(f"{Location_plots}/13_power_factor.png")
        plt.close(fig13)


        # -------------------------------------------------
        # Figure 14: Phase angle
        # -------------------------------------------------

        fig14, ax14 = plt.subplots(figsize=(6.4, 4.8))
        ax14.plot(np.arange(0, len(df_3["pf"]), 1), np.degrees(df_3["phi"]), label="Phase angle")
        ax14.set_xlabel("Time (s)")
        ax14.set_ylabel("Degrees (°)")
        ax14.set_title("Phase angle")
        ax14.set_xlim(min(np.arange(0, len(df_3["pf"]), 1)), max(np.arange(0, len(df_3["pf"]), 1)))
        ax14.legend(loc="best")
        ax14.grid(True)
        plt.savefig(f"{Location_plots}/14_phase_angle.png")
        plt.close(fig14)


        # -------------------------------------------------
        # Figure 15: Inverter rms voltage and current
        # -------------------------------------------------

        fig15, ax15 = plt.subplots(figsize=(6.4, 4.8))
        ax15.plot(np.arange(0, len(df_3["pf"]), 1), df_3["Vs"], label="Inverter RMS voltage (V)")
        ax15.plot(np.arange(0, len(df_3["pf"]), 1), df_3["Is"], label="Inverter RMS current (A)")
        ax15.set_xlabel("Time (s)")
        ax15.set_ylabel("Value")
        ax15.set_title("RMS voltage and current")
        ax15.set_xlim(min(np.arange(0, len(df_3["pf"]), 1)), max(np.arange(0, len(df_3["pf"]), 1)))
        ax15.legend(loc="best")
        ax15.grid(True)
        plt.savefig(f"{Location_plots}/15_inverter_rms_voltage_and_current.png")
        plt.close(fig15)


        # -------------------------------------------------
        # Figure 16: Inverter DC voltage
        # -------------------------------------------------

        fig16, ax16 = plt.subplots(figsize=(6.4, 4.8))
        ax16.plot(np.arange(0, len(df_3["pf"]), 1), df_3["V_dc"], label="Inverter DC voltage")
        ax16.set_xlabel("Time (s)")
        ax16.set_ylabel("Voltage (V)")
        ax16.set_title("Inverter DC voltage")
        ax16.set_xlim(min(np.arange(0, len(df_3["pf"]), 1)), max(np.arange(0, len(df_3["pf"]), 1)))
        ax16.legend(loc="best")
        ax16.grid(True)
        plt.savefig(f"{Location_plots}/16_inverter_dc_voltage.png")
        plt.close(fig16)


        # -------------------------------------------------
        # Figure 17: IGBT and Diode heat cycles
        # -------------------------------------------------

        fig17, ax17 = plt.subplots(figsize=(6.4, 4.8))
        ax17.plot(df_2["time_period_df2"], df_2["t_cycle_heat_I"], label="IGBT",linewidth=5)
        ax17.plot(df_2["time_period_df2"], df_2["t_cycle_heat_D"], label="Diode")
        ax17.set_xlabel("Time (s)")
        ax17.set_ylabel("Time (s)")
        ax17.set_title("IGBT and Diode heat cycles timing")
        ax17.set_xlim(min(df_2["time_period_df2"]), max(df_2["time_period_df2"]))
        ax17.legend(loc="best")
        ax17.grid(True)
        plt.savefig(f"{Location_plots}/17_IGBT_and_Diode_heat_cycles.png")
        plt.close(fig17)


        # -------------------------------------------------
        # Figure 18: IGBT and Diode Life (Dual Y-Axis, Log Scale)
        # -------------------------------------------------

        fig18, ax18 = plt.subplots(figsize=(6.4, 4.8))

        # Left axis (IGBT)
        ax18.bar("IGBT", df_2["Life_I"].iloc[0])
        ax18.set_ylabel("Time (years)" )
        ax18.set_title("IGBT and diode life period")
        #ax18.grid(axis="y", which="both")

        # Right axis (Diode)
        ax18_right = ax18.twinx()
        ax18_right.bar("Diode", df_2["Life_D"].iloc[0])
        ax18_right.set_ylabel("Time (years)")
        #ax18_right.yaxis.grid(True, which="both")  # right axis grid

        plt.savefig(f"{Location_plots}/18_Life_IGBT_and_Diode.png")
        plt.close(fig18)


        # -------------------------------------------------
        # Figure 19: Switch life
        # -------------------------------------------------

        fig19, ax19 = plt.subplots(figsize=(6.4, 4.8))
        ax19.bar("Switch", df_2["Life_switch"].iloc[0], width=0.4)  # width < 0.8 = thinner bar
        ax19.set_ylabel("Time (years)" )
        ax19.set_title("Switch life period")
        ax19.grid(True)
        plt.tight_layout()
        plt.savefig(f"{Location_plots}/19_Life_switch.png")
        plt.close(fig19)


        # -------------------------------------------------
        # Fig.20: Lifetime distribution of IGBT
        # -------------------------------------------------

        fig20, ax20 = plt.subplots(figsize=(6.4, 4.8))
        ax20.hist(df_4["Life_period_I_normal_distribution"],
                  bins=50,
                  edgecolor='black',
                  weights=np.ones(len(df_4["Life_period_I_normal_distribution"])) / len(df_4["Life_period_I_normal_distribution"]) * 100,
                  label="IGBT")

        # Axis labels & title
        ax20.set_xlabel("Time (years)")
        ax20.set_ylabel("Lifetime distribution (%)")
        ax20.set_title("Lifetime distribution of IGBT")
        ax20.grid(True, linestyle='--', alpha=0.6)
        ax20.legend()
        plt.savefig(f"{Location_plots}/20_Life_period_IGBT_normal_distribution.png")
        plt.close(fig20)


        # -------------------------------------------------
        # Fig.21: Lifetime distribution of Diode
        # -------------------------------------------------

        fig21, ax21 = plt.subplots(figsize=(6.4, 4.8))
        ax21.hist(df_4["Life_period_D_normal_distribution"],
                  bins=50,
                  edgecolor='black',
                  weights=np.ones(len(df_4["Life_period_D_normal_distribution"])) / len(df_4["Life_period_D_normal_distribution"]) * 100,
                  label="Diode")
        ax21.set_xlabel("Time (years)")
        ax21.set_ylabel("Lifetime distribution (%)")
        ax21.set_title("Lifetime distribution of diode")
        ax21.grid(True, linestyle='--', alpha=0.6)
        ax21.legend()
        plt.savefig(f"{Location_plots}/21_Life_period_Diode_normal_distribution.png")
        plt.close(fig21)


        # -------------------------------------------------
        # Fig.22: Lifetime distribution of Switch
        # -------------------------------------------------

        fig22, ax22 = plt.subplots(figsize=(6.4, 4.8))
        ax22.hist(df_4["Life_period_switch_normal_distribution"],
                      bins=50,
                      edgecolor='black',
                      weights=np.ones(len(df_4["Life_period_switch_normal_distribution"])) / len(df_4["Life_period_switch_normal_distribution"]) * 100,
                  label="Switch")
        ax22.set_xlabel("Time (years)")
        ax22.set_ylabel("Lifetime distribution (%)")
        ax22.set_title("Lifetime distribution of switch")
        ax22.grid(True, linestyle='--', alpha=0.6)
        ax22.legend()
        plt.savefig(f"{Location_plots}/22_Life_period_switch_normal_distribution.png")
        plt.close(fig22)


        # -------------------------------------------------
        # Fig.23: Cumulative distribution function of IGBT lifetime
        # -------------------------------------------------

        fig23, ax23 = plt.subplots(figsize=(6.4, 4.8))
        Calculation_functions_class.cdf_with_B_lines(ax23,
                          samples=np.asarray(df_4["Life_period_I_normal_distribution"]),
                          label="IGBT",
                          title="Cumulative distribution function of IGBT lifetime")
        plt.savefig(f"{Location_plots}/23_CDF_Life_period_IGBT.png")
        plt.close(fig23)


        # -------------------------------------------------
        # Fig.24: Cumulative distribution function of diode lifetime
        # -------------------------------------------------

        fig24, ax24 = plt.subplots(figsize=(6.4, 4.8))
        Calculation_functions_class.cdf_with_B_lines(ax24,
                          samples=np.asarray(df_4["Life_period_D_normal_distribution"]),
                          label="Diode",
                          title="Cumulative distribution function of diode lifetime")
        plt.savefig(f"{Location_plots}/24_CDF_Life_period_Diode.png")
        plt.close(fig24)


        # -------------------------------------------------
        # Fig.25: Cumulative distribution function of switch lifetime
        # -------------------------------------------------

        fig25, ax25 = plt.subplots(figsize=(6.4, 4.8))
        Calculation_functions_class.cdf_with_B_lines(ax25,
                          samples=np.asarray(df_4["Life_period_switch_normal_distribution"]),
                          label="Switch",
                          title="Cumulative distribution function of switch lifetime")
        plt.savefig(f"{Location_plots}/25_CDF_Life_period_Switch.png")
        plt.close(fig25)