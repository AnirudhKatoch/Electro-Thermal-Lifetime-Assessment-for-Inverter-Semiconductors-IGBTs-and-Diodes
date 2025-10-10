import subprocess
import sys

def run_once(entry_script, profile):
    """
    Run a given entry script as a completely new Python process.
    When the process exits, all its memory is freed.
    """
    print(f"\n🚀 Running {entry_script} for profile {profile}")
    cmd = [sys.executable, "-u", entry_script, "--profile", profile]
    subprocess.run(cmd, check=True)
    print("✅ Completed.\n")


if __name__ == "__main__":

    # Family reactive (Active already done)
    #run_once("main_1_entry_reactive.py", "synPRO_el_family_main_1")
    #run_once("main_2_entry_reactive.py", "synPRO_el_family_main_2_1_sec")
    #run_once("main_2_entry_reactive.py", "synPRO_el_family_main_2_15_min")

    ####################################################################################################################

    #Factor 1 alle active

    run_once("main_1_entry_active.py", "alle_main_2_1_inverter_1")
    run_once("main_2_entry_active.py", "alle_main_2_1_sec_inverter_1")
    run_once("main_2_entry_active.py", "alle_main_2_15_min_inverter_1")

    # Factor 1 alle Reactive

    run_once("main_1_entry_reactive.py", "alle_main_2_1_inverter_1")
    run_once("main_2_entry_reactive.py", "alle_main_2_1_sec_inverter_1")
    run_once("main_2_entry_reactive.py", "alle_main_2_15_min_inverter_1")

    ####################################################################################################################

    # Factor 1.5 alle active

    run_once("main_1_entry_active.py", "alle_main_2_1_inverter_1.5")
    run_once("main_2_entry_active.py", "alle_main_2_1_sec_inverter_1.5")
    run_once("main_2_entry_active.py", "alle_main_2_15_min_inverter_1.5")

    # Factor 1.5 alle reactive

    run_once("main_1_entry_reactive.py", "alle_main_2_1_inverter_1.5")
    run_once("main_2_entry_reactive.py", "alle_main_2_1_sec_inverter_1.5")
    run_once("main_2_entry_reactive.py", "alle_main_2_15_min_inverter_1.5")

    ####################################################################################################################

    # Factor 2 alle active

    run_once("main_1_entry_active.py", "alle_main_2_1_inverter_2")
    run_once("main_2_entry_active.py", "alle_main_2_1_sec_inverter_2")
    run_once("main_2_entry_active.py", "alle_main_2_15_min_inverter_2")

    # Factor 2 alle reactive

    run_once("main_1_entry_reactive.py", "alle_main_2_1_inverter_2")
    run_once("main_2_entry_reactive.py", "alle_main_2_1_sec_inverter_2")
    run_once("main_2_entry_reactive.py", "alle_main_2_15_min_inverter_2")