import numpy as np
from Calculation_functions_file import Calculation_functions_class

Calculation_functions_class = Calculation_functions_class()

class Input_parameters_class:

    def __init__(self,P=None,pf=None,Q=None):

        self.saving_dataframes = True # Set True if you want to save dataframes and False if you don't want to save dataframes.
        self.plotting_values = True   # Set True if you want to plot values and False if you don't want to plot values.

        self.chunk_seconds = int(86400)

        self.design_control = "switch" # "inverter" or "switch" , choose between two on what is the designing process,
        self.N_parallel = 50
        # In "switch" you can directly give in power values for the switch
        # In "inverter" if your power requirements are above the rated power of the switch the system will automatically put switches in parallel to match the power requirements.
        self.overshoot_margin_inverter = 0

        if P is None or pf is None or Q is None:
            self.P = np.full(86400, 30000.0, dtype=np.float64)
            self.pf =np.full(len(self.P), 1, dtype=float)   # [-] power factor [Inductive is negative and capacitive is positive]
            self.Q = np.full(len(self.P), 34500,dtype=float)  # [VAr] Inverter RMS Reactive power [Always give absolute values]
        else:
            self.P = np.asarray(P, dtype=float)
            self.pf = np.asarray(pf, dtype=float)
            self.Q = np.asarray(Q, dtype=float)

        self.Vs =np.full(len(self.pf), 230)     # [V] Inverter phase RMS AC side voltage
        #self.Vs = np.array([])                          # [V] Inverter RMS AC side voltage
        self.V_dc = np.full(len(self.pf), 545)   # [V] Inverter DC side voltage

        self.f = 50                                      # [Hz] Grid frequency
        self.M = 1.034                                   # [-] Inverter modulation index # Modulation cannot be above 1 as model does not take into account. Here I have done it barely to make the system follow physics law.
        self.Tamb = 298.15                               # [K] Ambient Temperature
        self.dt = 0.002                                  # [s] Simulation timestep (2 ms)

        if (self.pf[0] == 0 and self.Q[0] == 0):
            raise ValueError(
                "Invalid input: pf[0] = 0 and Q[0] = 0."
                "When the power factor is zero, you must provide a nonzero Q[0] "
                "(reactive power).")

        elif (self.pf[0] != 0 and self.P[0] == 0):
            raise ValueError(
                "Invalid input: pf[0] ≠ 0 but P[0] = 0."
                "When the power factor is nonzero, you must provide a nonzero P[0] "
                "(active power).")

        self.thermal_states = "shared"  # This defines the thermal state. If this is "separated" then IGBT and Diode have different paste and heat sink, if this is "shared" then IGBT and Diode are on the same paste and heat sink.
        if self.thermal_states not in ("separated", "shared"):
            raise ValueError("thermal_states must be 'separated' or 'shared'")

        self.inverter_phases = 3  # 1 or 3 (single-phase or three-phase)
        if self.inverter_phases not in (1, 3):
            raise ValueError("phases must be 1 or 3")

        # User option for modulation
        self.modulation_scheme = "svm"  # options: "spwm" or "svm" , the type of modulation once can choose for three phase inverters."svm" is  Space Vector PWM (or Third-Harmonic Injection) and "spwm" is Sinusoidal PWM (reference = pure sine).
        if self.modulation_scheme not in ("spwm", "svm"):  # when inverter_phases == 1 this variable is invalid.
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'")

        self.single_phase_inverter_topology = "full"  # options: "half" or "full"  # One can choose is the single phase inverter half bridge or full bridge
        if self.single_phase_inverter_topology not in ("half", "full"): # when inverter_phases == 3 this variable is invalid.
            raise ValueError("single_phase_inverter_topology must be 'half' or 'full'")

        self.Location_dataframes = "dataframe_files"

        # ----------------------------------------#
        # Switch Max limit
        # ----------------------------------------#

        self.max_V_CE = 600                        # [V] Maximum collector–emitter voltage
        self.max_IGBT_RMS_Current = 50             # [A] Maximum IGBT RMS Current
        self.max_IGBT_peak_Current = 200           # [A] Maximum IGBT Peak Current
        self.max_Diode_RMS_Current = 50            # [A] Maximum Diode RMS Current
        self.max_Diode_peak_Current = 200          # [A] Maximum Diode Peak Current
        self.max_IGBT_temperature = 448.15         # [K] Maximum IGBT Temperature
        self.max_Diode_temperature = 448.15        # [K] Maximum Diode Temperature
        self.overshoot_margin = 0.1                # [-] Overshoot margin

        # ----------------------------------------#
        # Max lifetime
        # ----------------------------------------#

        self.IGBT_max_lifetime = 30  # years
        self.Diode_max_lifetime = 30  # years

        # ----------------------------------------#
        # Switching losses
        # ----------------------------------------#

        # IGBT

        self.f_sw = 10 * 1000   # [Hz] Inverter switching frequency
        self.t_on = 60e-9       # [s] Effective turn-on time = td(on) + tr ≈ 23 ns + 37 ns (td is delay period and tr is rising time)  [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 5 datasheet]
        self.t_off = 259e-9     # [s] Effective turn-off time = td(off) + tf ≈ 235 ns + 24 ns (td is delay period and tf is fall time) [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 5 datasheet]

        # Diode

        self.I_ref = 30.0       # [A] Reference test current for diode reverse recovery  [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 6 datasheet]
        self.V_ref = 400.0      # [V] Reference test voltage for diode reverse recovery  [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 6 datasheet]
        self.Err_D = 0.352e-3   # [J] Reverse recovery energy per switching event       [Note: Value is temperature dependent, author assumes constant temp of 25°C] [Page 5 datasheet]

        # ----------------------------------------#
        # Conduction losses
        # ----------------------------------------#

        # IGBT

        self.R_IGBT = 0.01466  # [Ohm] Effective on-resistance for conduction model  [Note: Value is temperature and current dependent, author assumes constant current of 50 A and temp of 25°C] [Fig 6 and Page 5 datasheet]
        self.V_0_IGBT = 1.117  # [V]   Effective knee voltage                       [Note: Value is temperature and current dependent, author assumes constant current of 50 A and temp of 25°C] [Fig 6 and Page 5 datasheet]

        # Diode

        self.V_0_D = 1.23      # [V]    Effective forward knee voltage [Note: Value is temperature and current dependent, author assumes constant current of 30 A and temp of 25°C] [Fig 28 and Page 5 datasheet]
        self.R_D = 0.0164      # [Ohm]  Effective dynamic resistance   [Note: Value is temperature and current dependent, author assumes constant current of 30 A and temp of 25°C] [Fig 28 and Page 5 datasheet]

        # ----------------------------------------#
        # Thermal Parameters
        # ----------------------------------------#

        # Paste

        # Case-to-sink thermal interface (SIL PAD® TSP-1600)
        # Source: Henkel/Bergquist SIL PAD® TSP-1600 datasheet (TIM)

        self.r_paste = np.array([0.10])       # [K/W] Thermal resistance
        self.tau_paste = np.array([1e-4])     # [s]   Thermal time constant

        # Heat Sink

        # Sink-to-ambient (Aavid/Boyd 6399B, RθSA ≈ 3.3 K/W natural convection)
        # Source: Aavid/Boyd 6399B heatsink datasheet (TO-247 natural convection)

        self.r_sink = np.array([1.3, 2.0])    # [K/W] Thermal resistance
        self.tau_sink = np.array([0.8, 40.0]) # [s]   Thermal time constant

        # IGBT

        # Source: Infineon IKW50N60H3 datasheet, Fig. 21 (Foster RC coefficients)

        self.r_I = np.array([7.0e-3, 3.736e-2, 9.205e-2, 1.2996e-1, 1.8355e-1])                # [K/W] Thermal resistance
        self.tau_I = np.array([4.4e-5, 1.0e-4, 7.2e-4, 8.3e-3, 7.425e-2])                      # [s] Thermal time constant

        # Diode

        # Source: Infineon IKW50N60H3 datasheet, Fig. 22 (Foster RC coefficients)

        self.r_D = np.array([4.915956e-2, 2.254532e-1, 3.125229e-1, 2.677344e-1, 1.951733e-1])  # [K/W] Thermal resistance
        self.tau_D = np.array([7.5e-6, 2.2e-4, 2.3e-3, 1.546046e-2, 1.078904e-1])               # [s]   Thermal time constant

        # ----------------------------------------#
        # Inputs for calculating cycles
        # ----------------------------------------#

        self.A = 3.4368e14                     # [-]
        self.alpha = -4.923                    # [-] 5 K ≤ ΔT_junc ≤ 80 K       # This condition will not be satisfied as junction temperature will never go this low. the author will ignore this condition.
        self.beta1 = - 9.012e-3                  # [-]
        self.beta0 = 1.942                     # [-] 0.19 ≤ ar ≤ 0.42           #
        self.C = 1.434                         # [-]
        self.gamma = -1.208                    # [-] 0.07 s ≤ th ≤ 63 s         # As t_on
        self.fd = 0.6204                       # [-]  Diode
        self.fI = 1                            # [-]  IGBT
        self.Ea = 0.06606 * 1.60218e-19        # [J] 32.5 °C ≤ T_junc ≤ 122 °C  # This condition will be met most of the time but sometimes it will deviate. the author will ignore this condition.
        self.k_b = 8.6173324e-5 * 1.60218e-19  # [J/K]
        self.ar = 0.31                         # [-] Assuming to be of value of 0.31 but it will be different for every IGBT and diode

        # ----------------------------------------#
        # Miscellaneous
        # ----------------------------------------#

        self.t_cycle_heat_my_value = 0.005    # Heat cycle input by user.

        # --- Normalize all Foster arrays to fixed length N ---
        N_FOSTER = 5

        self.r_I, self.tau_I = Calculation_functions_class.resize_foster_branches(self.r_I, self.tau_I, N_FOSTER)
        self.r_D, self.tau_D = Calculation_functions_class.resize_foster_branches(self.r_D, self.tau_D, N_FOSTER)
        self.r_paste, self.tau_paste = Calculation_functions_class.resize_foster_branches(self.r_paste, self.tau_paste, N_FOSTER)
        self.r_sink, self.tau_sink = Calculation_functions_class.resize_foster_branches(self.r_sink, self.tau_sink, N_FOSTER)

    # ----------------------------------------#
    # Thermal state & constants
    # ----------------------------------------#

    # Precompute alphas (discrete-time decay factors for RC branches)
    @property
    def alpha_I(self):
        return np.exp(-self.dt / self.tau_I)

    @property
    def alpha_D(self):
        return np.exp(-self.dt / self.tau_D)

    @property
    def alpha_p(self):
        return np.exp(-self.dt / self.tau_paste)

    @property
    def alpha_s(self):
        return np.exp(-self.dt / self.tau_sink)

    # ----------------------------------------#
    # Miscellaneous
    # ----------------------------------------#

    @property
    def omega(self):       # [rad/s] Angular frequency of the grid (ω = 2πf)
        return 2 * np.pi * self.f

    @property
    def Time_period(self): # [-]  Number of seconds in simulation (length of pf array)
        return len(self.pf)

    @property
    def Nsec(self):        # [-] Number of simulation steps per second     (with dt = 1 ms → 1000 steps per second)
        return int(round(1.0 / self.dt))

    @property
    def Tgrid(self):       # [s] Grid period (time for one full cycle at frequency f)
        return 1.0 / self.f

    @property
    def Ngrid(self):       # [-] Number of simulation steps per grid cycle (at f = 50 Hz and dt = 2 ms → 10 steps per cycle)
        return int(round( self.Tgrid / self.dt))



