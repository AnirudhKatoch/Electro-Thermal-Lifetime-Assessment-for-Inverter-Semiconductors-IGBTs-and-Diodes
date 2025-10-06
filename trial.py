@staticmethod
def _kernel_key(Vs, Is, phi, V_dc, pf, M, f_sw, t_on, t_off, I_ref, V_ref, Err_D, R_IGBT, V_0_IGBT, R_D, V_0_D, omega,
                dt):
    # round to stabilize float hashing
    def r(x): return float(np.round(x, 12))

    return (r(Vs), r(Is), r(phi), r(V_dc), r(pf), r(M), r(f_sw), r(t_on), r(t_off), r(I_ref), r(V_ref), r(Err_D),
            r(R_IGBT), r(V_0_IGBT), r(R_D), r(V_0_D), r(omega), r(dt))


# simple module-level cache (dict) to avoid imports of functools in this file if you prefer
_KERNEL_CACHE = {}


@staticmethod
def _build_kernel_one_second(Vs, Is, phi, V_dc, pf, M, f_sw, t_on, t_off, I_ref, V_ref, Err_D, R_IGBT, V_0_IGBT, R_D,
                             V_0_D, omega, dt):
    """
    Build 1-second time grid and all electrical/loss arrays.
    Returns: (t, m, is_I, is_D, P_I, P_D, P_leg, vs_inverter, is_inverter)
    """

    t = np.arange(0.0, 1.0, dt, dtype=np.float64)

    vs_inverter, is_inverter = Calculation_functions_class.Inverter_voltage_and_current(Vs=Vs, Is=Is, phi=phi, t=t,
                                                                                        omega=omega)
    m = Calculation_functions_class.Instantaneous_modulation(M=M, omega=omega, t=t, phi=phi)
    is_I, is_D = Calculation_functions_class.IGBT_and_diode_current(Is=Is, t=t, m=m, omega=omega)

    P_sw_I, P_sw_D = Calculation_functions_class.Switching_losses(V_dc=V_dc, is_I=is_I, t_on=t_on, t_off=t_off,
                                                                  f_sw=f_sw, is_D=is_D, I_ref=I_ref, V_ref=V_ref,
                                                                  Err_D=Err_D)
    P_con_I, P_con_D = Calculation_functions_class.Conduction_losses(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M,
                                                                     pf=pf, is_D=is_D, R_D=R_D, V_0_D=V_0_D)

    P_I = np.ascontiguousarray(np.maximum(P_sw_I + P_con_I, 0.0), dtype=np.float64)
    P_D = np.ascontiguousarray(np.maximum(P_sw_D + P_con_D, 0.0), dtype=np.float64)
    P_leg = np.ascontiguousarray(P_I + P_D, dtype=np.float64)

    return (t, m, is_I, is_D, P_I, P_D, P_leg, vs_inverter, is_inverter)
