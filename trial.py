def Lifecycle_calculation_acceleration_factor(Nf,pf,Component_max_lifetime):
    Nf = np.asarray(Nf, dtype=float)

    # --- Miner damage per set (include the events inside each pf step) ---
    # If events_per_step varies with time, make it an array shaped like Nf.
    damage_per_set = float(np.sum(1 / Nf))   # sum_i (events_in_step_i / Nf)

    # --- correct time scaling to sets/year ---
    sets_per_year    = (3600.0 * 24.0 * 365.0) / (float(len(pf)))

    floor_damage_per_set = 1.0 / (sets_per_year * Component_max_lifetime)

    effective_damage_per_set = damage_per_set + floor_damage_per_set # SO this is total damage

    target_damage_per_set = (1.0 / Component_max_lifetime) / sets_per_year

    AF_I  = effective_damage_per_set / target_damage_per_set

    Life_I = Component_max_lifetime / AF_I

    return Life_I