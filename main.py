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
    # Run each simulation sequentially
    run_once("main_1_entry.py", "synPRO_el_family_main_1")
    run_once("main_2_entry.py", "synPRO_el_family_main_2_1_sec")
    run_once("main_2_entry.py", "synPRO_el_family_main_2_15_min")
