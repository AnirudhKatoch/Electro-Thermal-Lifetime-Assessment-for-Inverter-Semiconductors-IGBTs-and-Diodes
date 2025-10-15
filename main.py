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


    run_once("main_2_entry_active.py", "alle_main_2_1_sec_inverter_1")

